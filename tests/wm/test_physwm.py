"""Tests for PhysWM -- the physics-grounded two-path world model.

These guard the design rules that make the experiment meaningful, not
just that the code runs:

1. the solver is frozen and parameter-free;
2. gradients reach the probe *through* the solver;
3. theta is a forward-pass probe output, never a free variable;
4. Path A is supervised on the dataset ground truth; Path B is supervised
   on Path A's detached prediction; the teacher edge cannot train the Path-A
   decoder, while the probe-input edge deliberately shapes the predictor;
5. the loss decomposes exactly as ``L_A + alpha*L_B + beta*L_cons``.
"""

import pytest
import torch

from stable_worldmodel.wm.physwm import (
    BENCHMARKS,
    EpisodeWindowDataset,
    Normalizer,
    assert_no_free_theta,
    build_physwm,
    build_solver,
    physwm_loss,
    pokeworld_episodes,
    theta_r2,
)
from stable_worldmodel.wm.physwm.solvers import SOLVERS

B, T = 4, 5


def make_cfg(bench='pokeworld', **overrides):
    cfg = {
        'benchmark': bench,
        'probe_source': 'predicted',
        'probe_frames': 'context',
        'solver_state_source': 'gt',
        'encoder': {
            'name': 'tiny_cnn',
            'image_size': 32,
            'patch_size': 16,
            'embed_dim': 32,
            'width': 16,
        },
        'action_encoder': {'emb_dim': 16},
        'predictor': {
            'hidden_dim': 32,
            'depth': 1,
            'heads': 2,
            'dim_head': 16,
            'mlp_dim': 32,
            'max_frames': 8,
        },
        'decoder': {'hidden_dim': 32, 'depth': 2},
        'probe': {'hidden_dim': 0, 'mode': 'episode'},
        'solver': {
            'name': BENCHMARKS[bench]['solver'],
            'dt': 0.01,
            'substeps': 2,
        },
        'data': {
            'num_episodes': 2,
            'episode_length': 8,
            'window': T,
        },
    }
    cfg.update(overrides)
    return cfg


def make_model(bench='pokeworld', **overrides):
    spec = BENCHMARKS[bench]
    return build_physwm(
        make_cfg(bench, **overrides),
        spec['state_dim'],
        spec['action_dim'],
    )


def make_batch(model, bench='pokeworld'):
    spec = BENCHMARKS[bench]
    torch.manual_seed(0)
    batch = {
        'pixels': torch.rand(B, T, 3, 32, 32),
        'state': torch.randn(B, T, spec['state_dim']),
        'action': torch.randn(B, T, spec['action_dim']),
    }
    # tactile observation, supplied separately from `state` so the model
    # never has to index the supervision target to get an input
    if spec.get('tactile_index') is not None:
        batch['touch'] = torch.rand(B, T).abs()
    return batch


# ---------------------------------------------------------------------
# 1. the solver is frozen and parameter-free
# ---------------------------------------------------------------------


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_solver_has_zero_learnable_parameters(name):
    solver = build_solver(name)
    assert list(solver.parameters()) == []
    solver.assert_frozen()


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_solver_bounds_are_buffers_not_parameters(name):
    """Physics constants must be buffers so no optimizer can touch them."""
    solver = build_solver(name)
    buffers = dict(solver.named_buffers())
    assert {'theta_lo', 'theta_hi', 'theta_nominal'} <= set(buffers)


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_bound_theta_respects_bounds(name):
    solver = build_solver(name)
    raw = torch.randn(32, solver.theta_dim) * 50  # extreme raw values
    theta = solver.bound_theta(raw)
    assert torch.all(theta >= solver.theta_lo)
    assert torch.all(theta <= solver.theta_hi)


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_raw_for_inverts_bound_theta(name):
    solver = build_solver(name)
    theta = solver.theta_nominal.unsqueeze(0)
    assert torch.allclose(
        solver.bound_theta(solver.raw_for(theta)), theta, atol=1e-4
    )


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_solver_is_differentiable_wrt_theta(name):
    solver = build_solver(name)
    state = torch.randn(3, 2, solver.state_dim)
    action = torch.randn(3, 2, solver.action_dim)
    raw = torch.zeros(3, solver.theta_dim, requires_grad=True)
    out = solver(state, action, solver.bound_theta(raw))
    out.sum().backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all()
    assert raw.grad.abs().sum() > 0


@pytest.mark.parametrize('name', sorted(SOLVERS))
def test_solver_accepts_per_episode_and_per_step_theta(name):
    solver = build_solver(name)
    state = torch.randn(3, 4, solver.state_dim)
    action = torch.randn(3, 4, solver.action_dim)
    per_ep = solver.bound_theta(torch.zeros(3, solver.theta_dim))
    per_step = solver.bound_theta(torch.zeros(3, 4, solver.theta_dim))
    a = solver(state, action, per_ep)
    b = solver(state, action, per_step)
    assert a.shape == b.shape == state.shape
    # a constant per-step theta must equal the per-episode result
    assert torch.allclose(a, b, atol=1e-6)


# ---------------------------------------------------------------------
# 2 & 3. theta is a probe output, and gradients flow through the solver
# ---------------------------------------------------------------------


def test_no_free_theta_parameter():
    model = make_model()
    assert_no_free_theta(model)
    names = [n for n, _ in model.named_parameters()]
    assert not any(n.split('.')[-1].lower() == 'theta' for n in names)


def test_assert_no_free_theta_catches_a_free_variable():
    """The invariant must actually fire when violated."""
    model = make_model()
    model.register_parameter('theta', torch.nn.Parameter(torch.zeros(3)))
    with pytest.raises(AssertionError, match='free theta'):
        assert_no_free_theta(model)


def test_gradients_reach_probe_through_frozen_solver():
    """Path B's only learning signal passes through the physics solver."""
    model = make_model()
    batch = make_batch(model)
    out = model(batch)
    # use L_B alone: its sole route to the probe is through the solver
    loss = (out['state_b'] - out['state_a'].detach()).pow(2).mean()
    loss.backward()
    grad = model.probe.net[-1].weight.grad
    assert grad is not None
    assert grad.abs().sum() > 0, 'no gradient reached the probe'


def test_probe_is_a_pure_function_of_the_latent():
    """theta must depend only on the observation, not on the target."""
    model = make_model().eval()
    batch = make_batch(model)
    with torch.no_grad():
        t1 = model.infer_theta(batch)
        batch['state'] = batch['state'] + 100.0  # move the targets
        t2 = model.infer_theta(batch)
    assert torch.allclose(t1, t2), 'theta changed when only targets changed'


def test_probe_reads_the_action_conditioned_latent():
    """Changing only the action must change theta in the proposed route."""
    model = make_model().eval()
    batch = make_batch(model)
    with torch.no_grad():
        t1 = model.infer_theta(batch)
        batch['action'] = batch['action'] + 0.5
        t2 = model.infer_theta(batch)
    assert not torch.allclose(t1, t2), 'theta ignored action conditioning'


def test_physics_loss_shapes_the_predictive_latent():
    """L_B must induce structure in the shared predictor, not only the probe."""
    model = make_model()
    out = model(make_batch(model))
    (out['state_b'] - out['state_a'].detach()).pow(2).mean().backward()
    grad = model.predictor.out_proj.weight.grad
    assert grad is not None and grad.abs().sum() > 0


def test_encoded_probe_source_is_an_explicit_pre_action_ablation():
    model = make_model(probe_source='encoded')
    out = model(make_batch(model))
    (out['state_b'] - out['state_a'].detach()).pow(2).mean().backward()
    grad = model.predictor.out_proj.weight.grad
    assert grad is None or grad.abs().sum() == 0


def test_historical_config_without_probe_source_keeps_old_route():
    cfg = make_cfg()
    del cfg['probe_source']
    spec = BENCHMARKS['pokeworld']
    model = build_physwm(cfg, spec['state_dim'], spec['action_dim'])
    assert model.probe_source == 'encoded'


def test_tactile_observation_reaches_the_probe():
    """The tactile channel must actually inform theta.

    Without it the pixels are identical across theta and the dynamics
    depend only on k/m and c/m, so the probe could never separate mass,
    stiffness and drag. If this passes vacuously the whole PokeWorld
    identifiability claim is empty.
    """
    model = make_model().eval()
    batch = make_batch(model)
    assert model.tactile_index is not None, 'pokeworld must expose touch'
    with torch.no_grad():
        t1 = model.infer_theta(batch)
        batch['touch'] = batch['touch'] + 1.0
        t2 = model.infer_theta(batch)
    assert not torch.allclose(t1, t2), 'theta ignored the tactile observation'


def test_tactile_token_adds_exactly_one_token_per_frame():
    model = make_model().eval()
    batch = make_batch(model)
    with torch.no_grad():
        z = model.encode(batch)
    assert z.shape[2] == model.encoder.num_tokens + 1


def test_visual_only_mode_excludes_unobservable_touch_from_the_objective():
    model = make_model(tactile={'enabled': False}).eval()
    assert model.tactile_index is None
    assert model.state_loss_mask.tolist() == [True, True, True, True, False]
    out = model(make_batch(model))
    out['state_a'] = out['target'].clone()
    out['state_b'] = out['target'].clone()
    out['state_a'][..., -1] += 100.0
    out['state_b'][..., -1] -= 100.0
    losses = physwm_loss(out)
    assert losses['loss_a'] == 0
    assert losses['loss_b'] == 0


def test_tactile_token_bypasses_pooling():
    """The tactile token must not be averaged in with the pixel tokens.

    Mean-pooling all N tokens gives the tactile reading a 1/N share of a
    linear probe's input -- ~6% with 16 pixel tokens -- while it is the
    only token carrying what separates stiffness from mass and drag. It
    gets its own pathway instead, so it is half the probe's input.
    """
    model = make_model()
    probe = model.probe
    assert probe.tactile_tokens == 1
    embed = model.encoder.embed_dim
    # pooled observation tokens + the tactile token, kept separate
    assert probe.net[0].in_features == 2 * embed
    # and the probe stays low-capacity
    assert probe.num_params < 2000


def test_probe_is_low_capacity_by_default():
    model = make_model()
    assert model.probe.num_params < 2000
    # default is a single linear map
    assert len(model.probe.net) == 1


def test_per_episode_theta_shape_and_per_step_option():
    model = make_model()
    batch = make_batch(model)
    assert model(batch)['theta'].shape == (B, model.solver.theta_dim)

    cfg = make_cfg()
    cfg['probe'] = {'hidden_dim': 8, 'mode': 'step'}
    spec = BENCHMARKS['pokeworld']
    step_model = build_physwm(cfg, spec['state_dim'], spec['action_dim'])
    theta = step_model(batch)['theta']
    assert theta.shape == (B, T - 1, step_model.solver.theta_dim)


# ---------------------------------------------------------------------
# 4. A regresses ground truth; B regresses detached A
# ---------------------------------------------------------------------


def test_loss_b_does_not_train_path_a_decoder():
    """L_B must carry no gradient into the Path A decoder head.

    Path B's target IS Path A's prediction, but it is detached first: this
    is the structural guarantee that fitting the physics path never trains
    the Path-A decoder through the teacher target. The shared predictor is
    deliberately trained through Path B's probe-input edge. If ``a`` were
    not detached, the decoder would also receive gradient here.
    """
    model = make_model()
    batch = make_batch(model)
    out = model(batch)
    loss_b = (out['state_b'] - out['state_a'].detach()).pow(2).mean()
    loss_b.backward()
    head = model.state_decoder.net[-1]
    assert head.weight.grad is None or head.weight.grad.abs().sum() == 0


def test_loss_a_does_not_depend_on_the_probe():
    """Symmetrically, L_A must not train the probe."""
    model = make_model()
    batch = make_batch(model)
    out = model(batch)
    loss_a = (out['state_a'] - out['target']).pow(2).mean()
    loss_a.backward()
    grad = model.probe.net[-1].weight.grad
    assert grad is None or grad.abs().sum() == 0


def test_a_regresses_dataset_b_regresses_path_a():
    """A's target is the dataset ``s_next``; B's target is A, detached."""
    model = make_model()
    out = model(make_batch(model))
    losses = physwm_loss(out)
    assert torch.allclose(
        losses['loss_a'], (out['state_a'] - out['target']).pow(2).mean()
    )
    assert torch.allclose(
        losses['loss_b'],
        (out['state_b'] - out['state_a'].detach()).pow(2).mean(),
    )
    # and NOT against the dataset target directly
    assert not torch.allclose(
        losses['loss_b'], (out['state_b'] - out['target']).pow(2).mean()
    )


def test_dataset_physics_target_is_only_an_explicit_ablation():
    model = make_model()
    out = model(make_batch(model))
    losses = physwm_loss(out, physics_target='dataset')
    assert torch.allclose(
        losses['loss_b'], (out['state_b'] - out['target']).pow(2).mean()
    )


def test_bad_physics_target_is_rejected():
    model = make_model()
    with pytest.raises(AssertionError, match='physics_target'):
        physwm_loss(model(make_batch(model)), physics_target='theta_labels')


# ---------------------------------------------------------------------
# 5. loss composition
# ---------------------------------------------------------------------


@pytest.mark.parametrize('alpha,beta', [(1.0, 0.0), (0.5, 0.1), (2.0, 1.0)])
def test_loss_decomposition_is_exact(alpha, beta):
    model = make_model()
    out = model(make_batch(model))
    losses = physwm_loss(out, alpha=alpha, beta=beta)
    expected = (
        losses['loss_a']
        + alpha * losses['loss_b']
        + beta * losses['loss_consistency']
    )
    assert torch.allclose(losses['loss'], expected)


def test_consistency_is_off_by_default():
    model = make_model()
    out = model(make_batch(model))
    losses = physwm_loss(out)
    assert torch.allclose(losses['loss'], losses['loss_a'] + losses['loss_b'])


def test_consistency_compares_the_two_paths():
    model = make_model()
    out = model(make_batch(model))
    losses = physwm_loss(out, beta=1.0)
    assert torch.allclose(
        losses['loss_consistency'],
        (out['state_a'] - out['state_b']).pow(2).mean(),
    )


def test_bad_consistency_detach_mode_is_rejected():
    model = make_model()
    out = model(make_batch(model))
    with pytest.raises(AssertionError):
        physwm_loss(out, consistency_detach='both')


# ---------------------------------------------------------------------
# model plumbing
# ---------------------------------------------------------------------


@pytest.mark.parametrize('bench', sorted(BENCHMARKS))
def test_forward_shapes_for_every_benchmark(bench):
    model = make_model(bench)
    batch = make_batch(model, bench)
    out = model(batch)
    S = BENCHMARKS[bench]['state_dim']
    assert out['state_a'].shape == (B, T - 1, S)
    assert out['state_b'].shape == (B, T - 1, S)
    assert out['target'].shape == (B, T - 1, S)


def test_solver_reports_zero_params_in_model_report():
    model = make_model()
    assert model.param_report()['solver'] == 0


def test_state_dim_mismatch_is_rejected():
    cfg = make_cfg('pokeworld')
    with pytest.raises(AssertionError, match='state_dim'):
        build_physwm(cfg, state_dim=99, action_dim=2)


def test_normalizer_round_trip():
    norm = Normalizer(3)
    x = torch.randn(100, 3) * 5 + 2
    norm.fit(x)
    assert torch.allclose(norm.unnorm(norm.norm(x)), x, atol=1e-4)
    assert bool(norm.fitted)


def test_predictor_is_block_causal():
    """A frame must not see the future -- else Path A can copy the answer."""
    model = make_model()
    mask = model.predictor.block_causal_mask(3, 2, torch.device('cpu'))
    # query in frame 0 (rows 0,1) may not attend to frames 1,2 (cols 2..5)
    assert not mask[0, 2:].any()
    assert mask[0, :2].all()
    # last frame sees everything
    assert mask[-1].all()


def test_legacy_probe_frames_drops_the_last_unscored_latent():
    """Legacy mode retains its historical T-1 temporal alignment."""
    model = make_model()
    batch = make_batch(model)
    assert model.probe_frames == 'context'
    z_probe = model.probe_latent(batch)
    assert z_probe.shape[1] == T - 1
    assert model.probe(z_probe).shape[0] == B


def test_context_query_split_hides_query_outcomes_from_episode_theta():
    model = make_model(
        probe_context=2, physics_loss_scope='context'
    ).eval()
    batch = make_batch(model)
    with torch.no_grad():
        theta_before = model.infer_theta(batch)
        # Query outcomes/future actions begin after frame/action index 2.
        batch['pixels'][:, 3:] = torch.rand_like(batch['pixels'][:, 3:])
        batch['action'][:, 3:] += 100.0
        batch['touch'][:, 3:] += 100.0
        theta_after = model.infer_theta(batch)
    assert torch.allclose(theta_before, theta_after, atol=1e-6)
    assert model.probe_latent(batch).shape[1] == 3  # indices 0..K
    loss_mask, eval_mask = model.physics_time_masks(T, torch.device('cpu'))
    assert loss_mask.tolist() == [True, True, False, False]
    assert eval_mask.tolist() == [
        False, False, True, True,
    ]


def test_physics_loss_uses_context_while_metrics_hold_out_query():
    model = make_model(probe_context=2, physics_loss_scope='context')
    out = model(make_batch(model))
    out['state_a'] = torch.zeros_like(out['state_a'])
    out['target'] = torch.zeros_like(out['target'])
    out['state_b'] = torch.zeros_like(out['state_b'])
    out['state_b'][:, 2:] = 100.0  # query errors are held out from training
    assert physwm_loss(out)['loss_b'] == 0
    out['state_b'][:, :2] = 1.0
    assert physwm_loss(out)['loss_b'] > 0


# ---------------------------------------------------------------------
# data
# ---------------------------------------------------------------------


def test_windows_never_straddle_episodes():
    episodes = [
        {
            'state': torch.arange(20).float().reshape(10, 2).numpy(),
            'action': torch.zeros(10, 2).numpy(),
        }
        for _ in range(3)
    ]
    ds = EpisodeWindowDataset(episodes, window=4)
    # every window must be contiguous within one episode
    for i in range(len(ds)):
        s = ds[i]['state'][:, 0]
        assert torch.allclose(s.diff(), torch.full((3,), 2.0))


def test_window_longer_than_episode_is_rejected():
    episodes = [
        {
            'state': torch.zeros(3, 2).numpy(),
            'action': torch.zeros(3, 2).numpy(),
        }
    ]
    with pytest.raises(AssertionError, match='no windows'):
        EpisodeWindowDataset(episodes, window=8)


def test_contact_conditioning_uses_context_only_not_query_outcome():
    episode = {
        'state': torch.zeros(7, 2).numpy(),
        'action': torch.zeros(7, 2).numpy(),
        # One contact at frame 2. With K=2 it is context for starts 0/1,
        # but not for start 2, whose first query outcome is frame 5.
        'touch': torch.tensor([0, 0, 1, 0, 0, 0, 0]).numpy(),
    }
    ds = EpisodeWindowDataset(
        [episode], window=5, context=2, require_context_touch=True
    )
    assert ds.index == [(0, 0), (0, 1)]


def test_pokeworld_exposes_ground_truth_theta():
    eps, sim = pokeworld_episodes(3, 12, seed=0, render=False)
    assert all('theta_true' in e for e in eps)
    thetas = torch.stack([torch.from_numpy(e['theta_true']) for e in eps])
    # theta varies across episodes -> R^2 is well defined
    assert thetas[:, :3].std(0).min() > 0
    assert len(sim.theta_names) == thetas.shape[1]


def test_theta_r2_is_one_for_a_perfect_probe():
    true = torch.randn(16, 3)
    r2 = theta_r2(true.clone(), true, ('a', 'b', 'c'))
    for v in r2.values():
        assert pytest.approx(1.0, abs=1e-5) == float(v)


def test_theta_r2_is_nan_for_a_constant_parameter():
    """R^2 is undefined when the true parameter does not vary.

    Regression guard: dividing by a collapsed variance used to produce
    absurd values like -3e14 instead of an honest 'undefined'.
    """
    true = torch.stack(
        [torch.randn(16), torch.full((16,), 0.1)], dim=1
    )  # second column is constant
    pred = true + 0.01
    r2 = theta_r2(pred, true, ('varies', 'constant'))
    assert torch.isfinite(torch.tensor(float(r2['r2/varies'])))
    assert torch.isnan(torch.tensor(float(r2['r2/constant'])))


def test_theta_r2_rejects_mismatched_shapes():
    with pytest.raises(AssertionError, match='shape mismatch'):
        theta_r2(torch.randn(8, 3), torch.randn(8, 4), ('a', 'b', 'c'))
