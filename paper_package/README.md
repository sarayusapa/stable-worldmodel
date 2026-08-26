# PhysWM paper-writing package

This folder is a self-contained handoff for writing a paper around the claim:

> **PhysWM grounds an unconstrained predictive world model through known physical equations by forcing its latent representation to expose a compact set of physical variables.**

Start with [`PAPER_HANDOFF.md`](PAPER_HANDOFF.md). It contains the proposed framing, paper-ready prose, methods, evaluation protocol, quantitative findings, limitations, and a claim/evidence boundary. [`EXPERIMENT_LEDGER.md`](EXPERIMENT_LEDGER.md) records which experiment families are safe to cite, which are diagnostic, and which remain incomplete.

The `generated/` directory contains PNG and SVG figures, CSV tables, and a machine-readable metric summary. Rebuild them with:

```bash
python paper_package/build_paper_assets.py \
  --runs-root /workspace/physwm-artifacts/runs
```

The plotting script deliberately excludes the pre-fix Fetch matrix. It uses these evidence groups:

- PokeWorld decodability: `pilot-2048ep`, three seeds.
- PokeWorld functional evaluation: `functional-2048`, three seeds.
- PushT functional evaluation: `pusht-cartpole-matrix/pusht_rand_xbench_*`, three seeds.

CartPole and corrected Fetch runs were incomplete when this package was created on 2026-08-26 at approximately 06:00 UTC. Refresh the ledger and regenerate assets before submission.

