# PhysWM: grounding predictive world models in physical equations

## One-sentence thesis

**PhysWM grounds an unconstrained predictive world model through known physical equations by forcing its latent representation to expose a compact set of physical variables.**

This is the central framing. The work is primarily about physics-grounded world modeling and equation-constrained representation learning. Parameter substitution is an evaluation tool, not the identity of the paper. Likewise, the paper should not lead with causal intervention, active exploration, planning, or physical-parameter recovery because the current method and evidence do not fully establish those broader claims.

## Candidate titles

1. **PhysWM: Grounding Predictive World Models in Physical Equations**
2. **From Neural Prediction to Physical Explanation: Equation-Grounded World Models**
3. **Physics as a Latent Constraint for Predictive World Models**
4. **Learning Equation-Compatible Representations without Physical Parameter Labels**

The first title is the clearest and matches the actual mechanism.

## Paper context and motivation

Learned world models have become powerful predictors of visual dynamics. Given a history of observations and actions, a neural predictor can forecast the next state or observation and can often support planning or control. Predictive accuracy, however, does not imply that the learned representation is organized around the physical variables that govern the system. A high-capacity model may exploit appearance, trajectory regularities, or policy correlations while leaving quantities such as mass, contact stiffness, damping, friction, and controller response entangled or inaccessible. This creates a gap between predicting what happens and representing why the dynamics behave as they do.

Physics-informed learning commonly addresses this gap by training directly against physical state or parameter labels, placing differential equations inside the learned transition function, or optimizing a simulator to reproduce observed trajectories. Those approaches are valuable when the governing model and supervision are sufficiently complete. They are less straightforward when a learned world model already provides a useful predictive representation, physical parameters are unavailable as annotations, and the available simulator is only an approximation of the environment. In that setting, replacing the neural world model with an analytic simulator can sacrifice predictive flexibility, while merely probing the neural latent for physical labels does not change how the representation is learned.

PhysWM instead treats known physical equations as a grounding constraint on an otherwise flexible predictive world model. The architecture produces two next-state predictions from the same action-conditioned latent. A neural decoder forms an unconstrained prediction, Path A, trained against the observed next state. A low-capacity probe maps the shared predictive latent to a compact vector of physical variables. A frozen differentiable solver then maps the current state, action, and inferred variables to an equation-based prediction, Path B. Rather than supervising the physical vector with parameter labels, PhysWM trains Path B to explain the stopped-gradient prediction produced by Path A. The neural model remains anchored to data, while the physical branch asks whether its prediction can be expressed through a known family of equations.

This formulation provides a middle ground between purely neural prediction and fully specified physical simulation. Path A can absorb residual effects, visual complexity, and simulator mismatch. Path B supplies an explicit physical coordinate system and a constrained explanatory route. The physical probe is deliberately low capacity—linear by default—so it cannot hide a second world model. The simulator has no learned parameters, and the inferred physical vector is produced by a forward pass rather than optimized separately for every test trajectory. Physical grounding must therefore be expressed in the shared predictive representation.

## Suggested abstract

Predictive world models can forecast dynamics without organizing their representations around the physical variables that govern a system. We introduce **PhysWM**, a two-path architecture that grounds an action-conditioned predictive world model through known physical equations. The first path is an unconstrained neural next-state predictor trained on observed transitions. The second maps the same predictive latent through a low-capacity probe to a compact physical vector and then through a frozen differentiable simulator. Without access to physical-parameter labels, the equation-based path is trained to reproduce the neural model's stopped-gradient prediction, encouraging the shared latent to support both flexible prediction and physical explanation. We evaluate the resulting representation through held-out prediction, supervised linear readout where parameter labels exist, parameter-substitution tests, and multi-horizon equation-based rollouts. On PokeWorld, the neural predictor consistently improves over persistence and inferred variables yield lower long-horizon rollout error than shuffled variables, although semantic recovery is strong for stiffness and weak for mass and drag. On randomized PushT dynamics, inferred effective variables outperform shuffled episode assignments across three seeds, while benefits over nominal parameters and predictive accuracy are not uniform. These results support equation-grounded effective dynamics while exposing two fundamental limitations: physical variables may be non-identifiable under partial observation, and inferred variables may compensate for simulator mismatch rather than equal ground-truth parameters. PhysWM therefore offers a practical route for grounding predictive world models in physical structure while making the boundary between physical explanation and effective system identification explicit.

## Main contributions

The contribution list should remain narrow and testable:

1. **A two-path physics-grounded world model.** A neural predictor and a frozen equation-based solver branch from the same action-conditioned latent, coupling flexible data-driven prediction with a structured physical explanation.
2. **Label-free induction of physical variables.** A low-capacity probe predicts bounded physical variables without physical-parameter supervision; the solver path learns from the neural path's stopped-gradient prediction.
3. **A functional evaluation protocol.** In addition to linear decodability, the evaluation substitutes inferred, shuffled, nominal, and—where observable—true variables into the physical solver and measures held-out and multi-horizon behavior.
4. **An empirical analysis of identifiability and simulator mismatch.** The experiments distinguish semantic parameter recovery from effective variables that explain dynamics despite incomplete observations or approximate equations.

Do not claim a general-purpose neural simulator, causal discovery, active system identification, deformable-object modeling, 3D/4D reconstruction, sim-to-real transfer, or improved planning unless new experiments are added.

## Method

### Problem setting

Consider an episodic controlled dynamical system with observations (o_t), actions (a_t), and a prediction state (s_t). The system contains episode-persistent physical properties θ, such as mass, stiffness, drag, friction, geometry, or controller gains. During training, the model observes trajectories but does not receive θ as a supervision target. Given a causal context of observations and actions, the goal is to predict future state and induce a compact variable \(\hat\theta\) that makes this prediction compatible with a known differentiable physical model.

The physical model is written

\[
S(s_t,a_t,\theta) \rightarrow s_{t+1},
\]

where \(S\) contains no learnable parameters. It may exactly match the data-generating system, as in the synthetic PokeWorld benchmark, or act as a structured approximation, as in PushT and FetchPush.

### Shared predictive representation

An encoder maps observations into latent tokens,

\[
z_t = E_\phi(o_t),
\]

and an action-conditioned temporal predictor processes a causal history,

\[
\hat z_t = P_\psi(z_{\leq t},a_{\leq t}).
\]

Both prediction paths consume exactly this predictive latent. This detail is crucial: grounding an encoder feature before action conditioning would not require the representation used for future prediction to expose physical structure. The repository includes this pre-action route as an ablation through `probe_source=encoded`; it is not the proposed method.

### Path A: flexible neural prediction

A learned decoder predicts the next normalized state:

\[
\hat s^A_{t+1}=D_\omega(\hat z_t).
\]

Path A is trained directly against the dataset transition,

\[
\mathcal L_A = \left\|N(\hat s^A_{t+1})-N(s_{t+1})\right\|_2^2,
\]

where \(N\) denotes per-dimension state normalization. This term anchors the model to observed dynamics and prevents the neural predictor from simply inheriting errors in the physical approximation.

### Path B: equation-based physical explanation

A bounded low-capacity probe pools causal predictive latents and infers an episode-level vector,

\[
\hat\theta=\operatorname{Bound}\!\left(\rho_\xi\left(\operatorname{Pool}(\{\hat z_t\}_{t\in\mathcal C})\right)\right).
\]

The default probe is linear. The physical prediction is

\[
\hat s^B_{t+1}=S(s_t,a_t,\hat\theta).
\]

PhysWM does not train \(\hat\theta\) against a parameter label. Instead, the physical path explains the prediction already produced by the neural model:

\[
\mathcal L_B = \left\|N(\hat s^B_{t+1})-operatorname{sg}\left(N(\hat s^A_{t+1})\right)\right\|_2^2.
\]

The stop-gradient makes Path A a fixed target along this edge. It prevents the physics loss from moving the neural decoder toward the simulator, but gradients still pass through the differentiable solver, physical probe, and shared predictor. Thus, the physical constraint can reshape the predictive representation without forcing the flexible prediction head to imitate an imperfect solver.

The complete default objective is

\[
\mathcal L=\mathcal L_A+\alpha\mathcal L_B.
\]

An optional symmetric consistency term exists in the implementation but has weight \(\beta=0\) by default and should not appear as part of the main method unless explicitly enabled in reported experiments.

### Causal context and leakage prevention

The inferred physical vector is episode-persistent. With an eight-frame window and the default context boundary, the probe reads only causal predictive latents from completed context transitions. Physical fitting uses transitions before the boundary; evaluation is reported on later query transitions. No future target frame or query outcome enters the identifier. Functional experiments use a longer window with eight context transitions and seven held-out rollout transitions. Unit tests perturb query outcomes and verify that \(\hat\theta\) remains unchanged.

### What “grounding” means here

Grounding is operational, not rhetorical. A representation is more physically grounded when a compact variable read from the predictive latent allows fixed physical equations to reproduce held-out dynamics better than variables that contain no episode-specific information. This does not require every coordinate of \(\hat\theta\) to equal a named ground-truth parameter. Under partial observability, multiple physical parameter combinations may induce the same trajectory. Under solver mismatch, the best explanatory variables may be effective parameters that compensate for omitted mechanisms. The evaluation therefore separates semantic recovery from functional equation compatibility.

## Evaluation protocol

### Question 1: Is the learned predictor meaningful?

Every representation claim is paired with the neural predictor's held-out normalized RMSE and a persistence baseline, \(\hat s_{t+1}=s_t\). If Path A does not beat persistence, its latent cannot support a strong claim about physically explaining a useful prediction.

### Question 2: Are physical quantities accessible?

Where ground-truth parameters exist, a supervised ridge or linear readout is fitted on the exact pooled predictive latent and evaluated on held-out episodes. The model's own unsupervised physical probe is reported separately. This is a semantic certificate, not the training objective. Negative \(R^2\) means that predicting the validation mean would be better than the readout.

### Question 3: Do the inferred variables matter inside the equations?

The functional test replaces the solver input with several parameter sources while preserving the initial state and action sequence:

- **Inferred:** \(\hat\theta\) produced for the correct episode.
- **Shuffled:** \(\hat\theta\) taken from another episode.
- **Nominal:** the solver's fixed default values.
- **True:** the recorded physical parameters, when available.

The paper should call this a **parameter-substitution evaluation**. It measures whether the inferred vector contains episode-specific information that is used by the equations. It is not active intervention in the environment.

### Question 4: Does grounding survive multi-step simulation?

Starting from a true state, the frozen solver is rolled forward under the held-out action sequence. Errors are accumulated over horizon for each parameter source. A gap that widens between inferred and shuffled variables provides stronger evidence than one-step decodability because incorrect dynamics compound over time.

### Question 5: What breaks the physical interpretation?

Solver sensitivity measures how much each predicted state changes when a parameter coordinate is perturbed. Coordinates with near-zero sensitivity cannot be functionally identified from that solver. Solver adequacy compares persistence, nominal parameters, true parameters, and free oracle fitting. A solver that cannot beat persistence should not be used to claim physical grounding.

## Benchmarks

### PokeWorld

PokeWorld is a synthetic contact system with known physical parameters and an exactly matched differentiable solver. Episode-specific properties include mass, contact stiffness, and drag. Physical appearance is held fixed, so the parameters must be inferred from dynamics rather than a static visual cue. The benchmark also exposes an identifiability issue: visual motion determines ratios such as stiffness over mass and drag over mass, not necessarily each raw parameter. A tactile contact signal can break part of this degeneracy. Consequently, raw-parameter recovery should be discussed only for the observation condition that makes those parameters identifiable.

### PushT

PushT provides contact-rich planar manipulation with randomized effective dynamics. Its solver is intentionally approximate: the T-shaped block is modeled using a simplified contact geometry and mobility model. Ground-truth values for all inferred solver coordinates are not available, so semantic \(R^2\) is not the central metric. The appropriate test is whether episode-specific inferred variables explain transitions better than shuffled and nominal variables.

### FetchPush

FetchPush tests robot-object interaction with mass and friction variation. The benchmark currently functions primarily as a solver-mismatch case study. The original solver contained a corrected numerical bug, and the completed post-fix functional run still shows inferred variables dramatically outperforming true variables. This indicates compensation for residual model mismatch rather than clean parameter recovery. Fetch should remain a limitation or effective-parameter result until corrected three-seed primary runs and solver-adequacy checks are complete.

### CartPole

CartPole provides a controlled-dynamics benchmark with interpretable parameters, but the current collection protocol experienced native MuJoCo/EGL failures and a render-context leak. The leak is fixed, yet the remaining eight-episode jobs are not adequate headline evidence.

## Results supported by current records

All reported uncertainty below is the sample standard deviation across three seeds. Exact per-seed values and machine-readable summaries are in `generated/`.

### PokeWorld predictive premise

Across the three 2048-episode runs, the neural predictor obtains a mean query RMSE of approximately **0.282**, compared with **0.383** for persistence. It beats persistence in every seed. This establishes that Path A contains a useful action-conditioned dynamics signal rather than merely copying the current state.

### PokeWorld semantic recovery is selective

A supervised linear readout from the predictive baseline recovers contact stiffness with mean held-out \(R^2\) around **0.754**; PhysWM reaches approximately **0.784**. Mass recovery is much weaker: approximately **0.241** for the predictive baseline and **0.237** for PhysWM. Drag has negative \(R^2\) for both methods, improving only from roughly **−0.191** to **−0.146**. These results do not support a broad claim that PhysWM makes all true physical parameters linearly identifiable. They show that equation grounding preserves or modestly improves the most observable parameter while other coordinates remain ambiguous.

### PokeWorld functional equation compatibility

In one-step substitution, the inferred variables obtain mean scaled RMSE of approximately **0.088**, compared with **0.118** for shuffled variables and **0.103** for nominal variables. The exactly matched true parameters have essentially zero error, confirming solver adequacy. The inferred result is better than shuffled in two of three seeds and better than nominal in two of three seeds, so the one-step evidence is positive but not uniform.

At horizon seven, inferred variables obtain mean scaled RMSE of approximately **0.193**, compared with **0.220** for shuffled variables. The inferred curve is below the shuffled curve in every seed at the final horizon. This is the clearest PokeWorld evidence that the representation contains episode-specific variables useful inside the governing equations, even though those variables do not perfectly recover every named parameter.

### PushT effective dynamics

Across three seeds, inferred variables obtain mean substitution RMSE of approximately **0.208**, versus **0.267** for shuffled variables and **0.221** for nominal parameters. Inferred variables beat shuffled variables in all three seeds and nominal variables in two of three. The consistent inferred-versus-shuffled gap supports episode-specific effective dynamics.

The predictive premise is weaker. Path A beats persistence in seeds 1 and 2, but fails in seed 0 (`0.620` versus `0.584`). Solver-sensitivity records also show that most measurable influence is concentrated in the agent proportional and derivative gains; other contact and geometry coordinates have much smaller effects. The paper should therefore claim equation-grounded effective controller dynamics on PushT, not comprehensive recovery of its contact physics.

### FetchPush failure analysis

The original Fetch matrix is invalid for positive claims because a gripper displacement intended once per transition was applied once per numerical substep. After the 10× overshoot was fixed, the completed functional seed still reports scaled one-step errors of `1.535` for inferred, `6.984` for true, `7.357` for shuffled, and `22.364` for nominal variables. All parameter sources produce the same solver-space task success of `0.03125`. The inferred vector is therefore acting as a compensating effective parameter under model mismatch. This is evidence for flexibility of the bottleneck, but against interpreting its coordinates as recovered true mass and friction.

## Interpretation

The empirical picture supports a specific claim. Known equations can shape a predictive latent into variables that are useful when returned to those equations. This functional benefit can exist even when supervised recovery of named parameters is weak. The distinction is expected: trajectories constrain dynamical effects, not necessarily a unique semantic parameterization. For example, motion may reveal stiffness-to-mass and drag-to-mass ratios while leaving their individual factors ambiguous. Likewise, an approximate contact solver may use one coordinate to compensate for an omitted physical mechanism.

This motivates describing \(\hat\theta\) as an **effective physical coordinate** unless semantic identifiability has been separately certified. “Physical” refers to how the variable is used—inside a fixed equation family—not an automatic guarantee that its numerical value equals a ground-truth material property. This distinction should be prominent rather than buried as a caveat; it is one of the most scientifically valuable aspects of the work.

The results also argue for functional evaluation. A high linear-probe \(R^2\) can show correlation but does not prove that the representation controls an equation-based prediction. Conversely, useful equation-compatible coordinates can emerge without clean semantic readout. Prediction error, semantic readout, solver sensitivity, parameter substitution, and multi-horizon rollout answer different questions and should be reported together.

## Limitations

1. **The physical equations are supplied.** PhysWM selects variables within a known solver family; it does not discover governing equations from scratch.
2. **Grounding is limited by solver adequacy.** If the physical solver omits important mechanisms or contains numerical errors, inferred variables may compensate for mismatch.
3. **Effective variables need not be true parameters.** Without an identifiability certificate, a useful inferred coordinate should not be interpreted semantically.
4. **Partial observability limits recovery.** Static images do not expose episode-persistent physics, and short motion histories may identify only parameter ratios.
5. **Current visual scale is diagnostic.** The cited completed results use a tiny CNN. A three-seed frozen DINOv2 result is needed for a strong claim about modern pretrained visual world models.
6. **Cross-benchmark evidence is incomplete.** Corrected Fetch primary runs and stable CartPole runs were incomplete when this handoff was prepared.
7. **No active identification.** The model conditions on actions but does not choose actions to reduce physical uncertainty.
8. **No closed-loop control result.** Solver-space rollout and task-success diagnostics do not establish improved planning, model-based RL, or real-environment execution.
9. **Small number of seeds.** Three seeds support descriptive means and sample deviations but limited statistical power.

## Recommended paper structure

1. **Introduction:** prediction does not guarantee physical organization; present equation grounding and the one-sentence thesis.
2. **Related work:** world models, physics-informed learning, differentiable simulation, system identification, representation identifiability. Avoid portraying the method as causal discovery.
3. **Method:** two paths, stopped teacher, shared latent, frozen solver, bounded low-capacity probe, context/query split.
4. **Evaluation principles:** predictor premise, solver adequacy, semantic readout, substitution, sensitivity, multi-horizon rollout.
5. **Experiments:** lead with matched PokeWorld, then approximate PushT, then use Fetch as a mismatch analysis. Include CartPole only after a stable rerun.
6. **Discussion:** effective versus semantic physical variables; observability and solver mismatch.
7. **Limitations and conclusion.**

## Figure plan and captions

### Figure 1 — Architecture

File: `generated/fig1_architecture.{png,svg}`

**Caption:** *PhysWM grounds a predictive world model through known physical equations. An encoder and action-conditioned predictor produce a shared predictive latent. Path A decodes a flexible next-state prediction and is trained on the observed transition. Path B maps the same latent through a low-capacity probe to bounded physical variables and a frozen differentiable solver. The physical path explains Path A's stopped-gradient prediction, allowing the equation loss to shape the shared representation without pulling the neural decoder toward an imperfect solver.*

### Figure 2 — PokeWorld functional grounding

File: `generated/fig2_pokeworld_functional.{png,svg}`

**Caption:** *Functional evaluation on PokeWorld across three seeds. Left: one-step scaled RMSE after substituting inferred, true, shuffled, or nominal parameters into the frozen solver. Right: multi-horizon equation-based rollout error. True parameters are nearly exact because the solver matches the generator. Inferred variables reduce final-horizon error relative to shuffled variables in all seeds, showing episode-specific equation compatibility, although one-step improvements are not uniform.*

### Figure 3 — Prediction and semantic recovery

File: `generated/fig3_pokeworld_prediction_decodability.{png,svg}`

**Caption:** *PokeWorld predictive premise and semantic readout. The neural predictor beats persistence across seeds. A supervised linear readout strongly recovers contact stiffness, while mass recovery is weak and drag recovery is below the validation-mean baseline. Equation grounding should therefore be interpreted as inducing useful effective physical coordinates rather than uniformly recovering every named parameter.*

### Figure 4 — PushT effective dynamics

File: `generated/fig4_pusht_effective_dynamics.{png,svg}`

**Caption:** *Label-free evaluation under approximate PushT physics. Inferred episode-specific variables consistently outperform shuffled assignments, indicating that they carry dynamics information used by the solver. Improvements over nominal parameters occur in two of three seeds, and the neural predictor fails to beat persistence in seed 0, limiting the strength of the cross-benchmark claim.*

## Claim–evidence boundary for the paper-writing agent

### Safe claims

- PhysWM uses known differentiable equations to constrain a shared predictive latent without physical-parameter labels.
- The neural and physical predictions branch from the same action-conditioned representation.
- PokeWorld Path A beats persistence across the recorded three 2048-episode seeds.
- PokeWorld inferred variables beat shuffled variables at the final seven-step horizon in all three seeds.
- PushT inferred effective variables beat shuffled assignments in all three seeds.
- Physical semantic recovery is parameter-dependent and limited by identifiability.
- Approximate solvers can produce effective compensating parameters rather than true physical parameters.

### Claims requiring careful qualification

- “PhysWM recovers physical parameters”: qualify as effective physical variables; only stiffness shows strong semantic recovery in the cited PokeWorld runs.
- “PhysWM improves prediction”: Path A is not uniformly better than persistence on PushT and physics grounding does not uniformly improve the neural prediction.
- “Generalizes across systems”: only PokeWorld and PushT currently provide usable multi-seed evidence, and they test different notions of correctness.

### Do not claim from current evidence

- Discovery of unknown equations or causal graphs.
- Active interaction or information-seeking action selection.
- Deformable-object, 3D, or 4D physical modeling.
- Robust true mass/friction recovery on FetchPush.
- Improved closed-loop planning, model-based RL, safety, sim-to-real, or deployment.
- Paper-scale DINOv2 results until those runs exist.

## Reproducibility language

The implementation normalizes prediction losses while evaluating the solver in physical units. The solver owns no learnable parameters, physical variables are forward-pass outputs rather than free optimized tensors, and the stopped teacher edge prevents the physical loss from updating the Path-A decoder through its target. Experiment outputs record encoder, context window, episode count, epochs, seed, condition, batch size, and train/validation window counts. The repository also contains invariance tests for frozen solver parameters, absence of free physical variables, gradient routing, and context/query leakage. Before submission, the exact git revision and dirty-tree state should be attached to every final run; the current repository was at commit `bf4e356` with a user modification to `stable_worldmodel/wm/physwm/data.py` when this package was generated.

## Short conclusion

PhysWM addresses a gap between predictive accuracy and physical organization. It retains a flexible neural world model while requiring the model's shared predictive representation to support a compact explanation through known physical equations. Current evidence shows that the resulting variables can improve equation-based rollouts relative to shuffled episode assignments, particularly over longer horizons, but do not uniformly recover named ground-truth parameters. The central conclusion is therefore not that equations make every latent semantically identifiable. It is that equation compatibility provides a concrete, functional form of physical grounding—and that evaluating this grounding exposes the roles of observability, identifiability, and simulator fidelity.

