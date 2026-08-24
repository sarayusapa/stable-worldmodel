# Run Log

Every run and its results — training runs, smoke suites, evaluations,
ablations, benchmarks.

**This log is paper material.** Numbers reported in the paper are traced back to
entries here, so an entry needs enough detail that someone else could reproduce
the run from it alone.

---

## Read this before you run anything

**This applies to every agent and every human working in this repo, not just
whoever wrote a given entry.**

1. **Every run gets an entry**, with its full details, at the time you get the
   results. A run that isn't logged didn't happen as far as the paper is
   concerned.
2. Log **failed and inconclusive runs too.** Negative results are results, and a
   run that crashed at epoch 12 is something the next person needs to know.
3. **Newest entries go at the top** of the Log section, directly under the
   `## Log` heading.
4. Never edit or delete a past entry to make it look better. If an entry turns
   out to be wrong, add a new entry that corrects it and link back.
5. **Never invent details.** If you don't know the commit, config, seed, or
   hardware a number came from, write `unknown` — do not guess. An entry with
   honest gaps is usable; an entry with fabricated metadata is worse than no
   entry, and in a paper it's a retraction.
6. Copy metric tables **verbatim** from the tool that produced them. Do not
   round, reorder, or "clean up" numbers.
7. Note the repo state — commit sha, and whether the tree was dirty. The same
   command on a different working tree is a different experiment.
8. Anything in **Open items** at the bottom is unfinished work on this log.
   Clear items as you resolve them.

You do **not** need to log repo changes that produced no run. Code changes live
in git; this file is for results.

### Entry template

Copy this block for each new run.

```markdown
### YYYY-MM-DD — <short title>

- **Command:** <exact command line, including env vars and overrides>
- **Config:** <config file(s) + overrides>
- **Seed(s):** <seed, or the list for a multirun>
- **Commit / working tree:** <sha, branch, clean or dirty + what was uncommitted>
- **Hardware:** <CPU/GPU, or `unknown`>
- **Data:** <benchmark(s), episodes, episode length>
- **Duration:** <wall clock, or `unknown`>
- **Status:** pass / fail / partial
- **Artifacts:** <checkpoint dir, W&B run id, history.json path>

<results table or numbers, verbatim>

- **Notes:** <interpretation, caveats, what to run next>
```

---

## Log

### 2026-08-24 — PhysWM smoke suite across all three benchmarks

- **Command:** unknown — presumed `MUJOCO_GL=egl python scripts/smoke/run_smoke.py`
- **Config:** unknown (smoke defaults are `--epochs 3 --episodes 8 --length 24`,
  benchmarks `pokeworld cartpole pusht`)
- **Seed(s):** unknown
- **Commit / working tree:** branch `phy-wm`, HEAD `14ac2f5` ("Document PhysWM
  in the top-level README"), tree **dirty** — modified `README.md`,
  `scripts/train/physwm.py`, `scripts/train/config/physwm.yaml`,
  `stable_worldmodel/wm/physwm/README.md`, plus untracked
  `scripts/train/config/hardware/`. Local was 1 commit behind `origin/phy-wm`
  (`cd29179`, "changed GT of loss to general latent prediction"), so **this run
  does not include that loss change.**
- **Hardware:** unknown
- **Data:** `pokeworld`, `cartpole`, `pusht` — episodes/length unknown
- **Duration:** unknown
- **Status:** pass (all runs reached their epoch budget)
- **Artifacts:** unknown

| Run                      | Epochs completed | Best validation loss |
| ------------------------ | ---------------- | -------------------- |
| `physwm_pokeworld`       | 3/3              | 0.4218               |
| `physwm_cartpole`        | 2/2              | 0.1437               |
| `physwm_pusht`           | 2/2              | 0.6804               |
| `smoke_r2`               | 4/4              | 2.2311               |
| `smoke_physwm_pokeworld` | 3/3              | 0.4218               |
| `smoke_physwm_cartpole`  | 3/3              | 1.5374               |
| `smoke_physwm_pusht`     | 3/3              | 0.7882               |

- **Notes:** **Not citable in the paper as it stands** — the command, config,
  seed, hardware, and whether the uncommitted training-loop changes were active
  are all still missing. These are smoke-scale runs (a handful of epochs on 8
  short episodes); they show the pipeline runs end to end, not that it works.
  Three things need explaining before the numbers mean anything:
  - `physwm_pokeworld` and `smoke_physwm_pokeworld` report an identical best
    val loss (0.4218) at the same epoch count, while `physwm_cartpole`
    (0.1437, 2/2) and `smoke_physwm_cartpole` (1.5374, 3/3) diverge sharply.
    Unclear whether the `physwm_*` and `smoke_physwm_*` rows are separate runs
    or the same runs reported twice under different names.
  - `physwm_cartpole` and `physwm_pusht` stopped at 2 epochs where their
    `smoke_*` counterparts ran 3. Reason unknown.
  - `smoke_r2` (2.2311) is a different quantity from the per-benchmark losses;
    do not read it on the same scale.

  Losses across different benchmarks are **not** comparable to each other —
  different data, different scales. Only compare a benchmark to itself.

---

## Open items

- [ ] Backfill the 2026-08-24 smoke entry: exact command, config overrides,
      seed, hardware, and whether the uncommitted training-loop changes were in
      the tree.
- [ ] Resolve whether `physwm_*` and `smoke_physwm_*` are distinct runs, and why
      the cartpole numbers diverge between them.
- [ ] Explain the 2-epoch vs 3-epoch discrepancy in the cartpole and pusht runs.
- [ ] Define what `smoke_r2` measures and record its units/scale here.
- [ ] Re-run the smoke suite after merging `origin/phy-wm` `cd29179` (the loss
      ground-truth change) — the numbers above predate it and will shift.
- [ ] Establish the seeds-per-config convention for paper runs (the `hardware`
      profiles are set up for `seed=0,1,2` multiruns).
