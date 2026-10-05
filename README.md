# How much autonomy does a sparse BCI intent channel need?

A simulation study of hierarchical latent-space planning for assistive manipulation.

Real non-invasive BCIs deliver a few noisy, discrete commands per decision. Steering a robot through that
channel directly takes dozens of commands per task. This repo asks a narrow question: **if the user only has to
confirm a high-level intent (which disc, which zone), how much of the rest can a learned hierarchical planner
fill in, and what does that save in user input?**

> **Honest framing.** The BCI signal here is **simulated**: a discrete command vocabulary with a confusion matrix
> parameterised by decoder accuracy. It is **not EEG**. The arm is cosmetic, grasping is a kinematic weld, and
> observations are low-dimensional states. There is no user study. See [`docs/WRITEUP.md`](docs/WRITEUP.md) for
> what that does and does not support, and what would change with real decoded signals.

## Headline results

Default scene (3 discs, 3 zones, K = 9 possible tasks), 100 episodes per cell, fresh scene seeds (12000),
95% Wilson intervals are in [`docs/results_table.md`](docs/results_table.md). Cells show *unblocked / blocked* scenes.

| decoder accuracy | learned planner | scripted oracle executor | direct low-level control | BCI commands per task (planner vs direct) |
|---|---|---|---|---|
| 0.5 | 0.67 / 0.63 | 0.87 / 0.87 | 0.15 / 0.24 | 5.0 vs 70 |
| 0.7 | 0.79 / 0.76 | 0.99 / 0.99 | 0.63 / 0.64 | 3.1 vs 60 |
| 0.8 | 0.76 / 0.74 | 0.96 / 0.96 | 0.84 / 0.84 | 2.4 vs 50 |
| 0.9 | 0.74 / 0.73 | 0.94 / 0.94 | 0.95 / 0.95 | 1.0 vs 40 |
| 1.0 | 0.80 / 0.79 | 1.00 / 1.00 | 1.00 / 1.00 | 1.0 vs 31 |

What the data support, and what they do not:

* **Supported:** the planner needs roughly **5-25x less user input** (2-4 bits per task vs 23-73 bits) and
  overtakes direct control in success rate when decoder accuracy drops to about 0.7 or below.
* **Not supported:** that it is more reliable at high accuracy. Above ~0.8 direct control succeeds more often
  (at 10-60x the command count), and the planner trails the scripted oracle everywhere (0.63-0.80 vs 0.87-1.00).
* **Planning depth matters:** success rises from 0.21 (one effect step) to 0.62 (two) to 0.77 (three) and then
  saturates (see ablations in the write-up).
* **Negative result:** the planner **fails at pushing** (about 5% success on heavy, push-only targets in the
  mixed-skill scene, 0% in a push-only world) while the oracle succeeds. Four attempted fixes did not help;
  they are documented, not hidden.

![success vs accuracy](docs/figures/success_vs_accuracy.png)
![user effort](docs/figures/effort_vs_success.png)

## Demos

`site/index.html` (one clip per command), `site/scenarios.html` (multi-task sessions with a live HUD) and
`site/scenarios_mixed.html` (5 discs, light ones grasped, heavy ones pushed). Every clip is a fixed-seed run and
shows its true outcome, **failures included**.

![filmstrip](docs/figures/filmstrip.png)

## Method in one paragraph

A CAVIN-inspired two-level latent model (Fang et al., CoRL 2019), trained on self-supervised interaction data:
an *effect code* `c` predicts what changes over a 20-step macro step (meta-dynamics); a *motion code* `z`
decodes the effect into low-level actions (action generator). Planning is CEM/MPC over effect-code sequences
with the learned state-conditioned prior, re-planning every 5 steps. A Bayesian confirmation protocol turns
repeated noisy commands into a committed goal. This is a state-based re-implementation, **not** a reproduction
of the original system (see the write-up for the differences).

## Quick start

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 scripts/01_test_env.py             # environment + oracle smoke test, writes demo clips
python3 scripts/02_collect_data.py         # 50k self-supervised windows (~2 min)
python3 scripts/03_train.py --beta-c 0.5 --beta-z 0.1 --tag b05    # ~3 min on CPU
python3 scripts/04_evaluate.py --ckpt checkpoints/cavin_lite_b05.pt --methods hier oracle direct \
    --acc 0.5 0.6 0.7 0.8 0.9 1.0 --K 3 6 9 --episodes 100 --scene-seed 12000 --tag final
python3 scripts/06_make_figures.py
python3 scripts/07_make_demo.py --ckpt checkpoints/cavin_lite_b05.pt
python3 scripts/08_make_scenarios.py --ckpt checkpoints/cavin_lite_b05.pt
```

Mixed-skill scene (select the config with an environment variable):

```bash
BCIHP_CONFIG=configs/mixed.yaml python3 scripts/02_collect_data.py
BCIHP_CONFIG=configs/mixed.yaml python3 scripts/03_train.py --beta-c 0.5 --beta-z 0.1 --tag mixed
BCIHP_CONFIG=configs/mixed.yaml python3 scripts/04_evaluate.py --ckpt checkpoints/cavin_lite_mixed.pt \
    --methods hier oracle direct --acc 0.6 0.8 1.0 --K 15 --episodes 100 --scene-seed 12000 --tag mixed_final
BCIHP_CONFIG=configs/mixed.yaml python3 scripts/09_breakdown.py --csv results/eval_mixed_final.csv --scene-seed 12000
```

## Layout

```
configs/    default.yaml (3 discs) and mixed.yaml (5 discs: 3 light = graspable, 2 heavy = push only)
src/bcihp/  env, commands (noisy intent channel + belief session), datagen, model, train, planner,
            baselines (oracle, direct control), eval, utils
scripts/    01 smoke test, 02 data, 03 train, 04 evaluate, 05 diagnose, 06 figures, 07-08 demos,
            09 per-skill breakdown, 10 latent-structure check
docs/       WRITEUP.md (method, results, limitations), results tables, figures
site/       static demo pages and clips
```

## Evaluation protocol

Design choices were made on pilot seeds (5000+); ablations on seeds 9000+; the headline sweep on fresh seeds
12000+ with 100 episodes per cell. Scenes are paired across methods. The one choice made after seeing the
ablation seeds (re-planning every 5 steps) was re-validated on the headline seeds.

## Limitations (short version)

Simulated decoder and simulated user; abstract manipulation; planner reliability below the oracle and unsuited
to real deployment; pushing unsolved by the learned planner; direct-control baseline fails on pushing by
construction; hand-designed cost shaping of unproven benefit; no user study. Details in the write-up.

## References

See the reference list in [`docs/WRITEUP.md`](docs/WRITEUP.md).
