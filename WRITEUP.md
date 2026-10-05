# How much autonomy does a sparse BCI intent channel need?
### A simulation study of hierarchical latent-space planning for assistive manipulation

*Status: course-project-scale simulation study; not peer reviewed. All numbers come from the scripts in this repo.*

## 1. Summary

Non-invasive BCIs offer a small vocabulary of noisy, discrete commands, so controlling a robot arm through them
directly is slow and tiring. I built a simulated tabletop (MuJoCo, CPU only) where a user must get one of several
discs into one of several zones, and measured three ways of using a noisy command channel:
(i) **direct low-level control** (four directions plus grip), (ii) a **learned hierarchical latent planner**
(CAVIN-inspired) that receives only a confirmed high-level intent, and (iii) a **scripted oracle executor** that
receives the same confirmed intent (an upper bound that isolates decoding errors from planning errors).

Main findings (default scene, K = 9 tasks, 100 episodes per cell, fresh scenes):

1. The hierarchical planner needs **1-5 BCI commands per task, against 31-71 for direct control**; in bits of
   user input (Wolpaw estimate) about 2-4 bits against 23-73.
2. Below a decoder accuracy of about 0.7 the planner also **succeeds more often** than direct control
   (0.67-0.79 vs 0.15-0.64); above about 0.8 direct control succeeds more often (0.84-1.00 vs 0.74-0.80).
3. The learned planner **trails the scripted oracle everywhere** (0.63-0.80 vs 0.87-1.00). Its benefit over a
   scripted executor is *not* demonstrated here; the study shows how much autonomy helps, not that learning it
   is necessary.
4. **Planning depth matters**: success rises from 0.21 to 0.62 to 0.77 for 1, 2, 3 effect steps and saturates.
5. In a mixed-skill scene (5 discs; light ones can be grasped, heavy ones must be pushed) the planner handles
   grasping (0.78-0.87) but **fails at pushing (about 5%)** while the oracle succeeds (0.84-1.00). Four attempts
   to fix this failed (Section 8).

## 2. Motivation and honest framing

The central constraint of non-invasive BCI control is information rate: a user can specify *what* they want at a
coarse level but not a motor trajectory. Shared control is the standard response (Section 3). This project
quantifies the trade-off in one controlled setting: **success and user input as a function of decoder accuracy
and vocabulary size, for different amounts of robot autonomy.**

What is simulated, explicitly:

* **The BCI channel.** A decoded command is drawn from a confusion matrix built from a single accuracy parameter
  (errors biased toward neighbouring command indices). No EEG is used. Real decoders have subject-specific,
  non-uniform, non-stationary error structure, latency, and session-to-session drift.
* **The user.** One fixed intended command per task, re-issued until the system commits. The user never tires, is
  never ambiguous, and in the direct-control baseline knows the optimal plan (an optimistic baseline).
* **The manipulation.** Planar discs on a table; the arm drawn above is cosmetic (a pusher moves by end-effector
  deltas); grasping is a kinematic weld ("magic grasp") within reach; observations are low-dimensional states,
  not images. "Heavy" discs are simply ungraspable; their pushing dynamics are identical to light ones.

## 3. Related work and gap analysis

*A non-exhaustive search (about ten queries, October 2026), not a systematic review. Items marked [verify] need
a check of authors or venue before this text is used in an application.*

**Shared control for BCI-driven robots is well established.** Reviews cover shared autonomy for cursors, arms and
rehabilitation robots (arXiv 2506.16044). Muelling et al. ("Autonomy infused teleoperation with application to BCI
manipulation") combine vision, user-intent inference and arbitration to assist intracortical-BCI users of a 7-DoF
arm. A two-class motor-imagery BCI with vision-guided autonomous grasping was shown for a dexterous arm
(*Robotics and Autonomous Systems*, vol. 115, 2019). Iturrate et al. used error-related potentials for
shared-control reaching (EMBC 2013) and observed ~80% single-trial ErrP classification with a real robot (EMBC 2010).
Douglas et al. compare levels of shared autonomy in brain-robot interfaces for activities of daily living [verify year].
In general shared autonomy, Javdani et al. (RSS 2015) formulate assistance as a POMDP with uncertainty over the user's goal.

**Task-level, hierarchical BCI control with noisy commands is also established.**
Chung, Cheung, Scherer and Rao (IJCAI 2011) propose a hierarchical BCI in which learned skills are invoked as
high-level commands, with a Gaussian-process uncertainty estimate switching between autonomous and user-guided
modes. "Task Level Hierarchical System for BCI-enabled Shared Autonomy" (Humanoids 2017, Columbia) lets a user give
high-level direction through a low-bandwidth SSVEP BCI while the robot handles navigation and grasping. Burget et al.
(ECMR 2017) connect BCI decoding to a high-level task planner that decomposes the selected task into atomic
actions. An ESANN 2009 paper on augmenting BCI information for assistive devices maintains a belief over user plan
hypotheses from noisy discrete commands [verify title and authors]. My confirmation protocol (a Bayesian belief over
intents, committing at a threshold) is a simple instance of this idea.

**Learned latent structure for assistive control.** Losey et al. learn latent actions that map a low-dimensional
human input (e.g. a 2-DoF joystick) to high-dimensional robot behaviour. That work targets continuous low-dimensional
input; mine targets discrete intents.

**Hierarchical latent planning in manipulation.** CAVIN (Fang, Zhu, Garg, Savarese, Fei-Fei, CoRL 2019) decouples
high-level effect prediction from low-level motion generation through cascaded variational inference and plans at two
temporal resolutions; its public code (StanfordVL/cavin) covers planar pushing tasks from raw visual observations.
Neural Task Programming (Xu et al., ICRA 2018) and Neural Task Graphs (Huang et al., CVPR 2019) infer hierarchical
task structure from demonstrations. These are not BCI works.

**What is left, stated modestly.**
The concept (low-bandwidth BCI plus autonomous low-level execution plus belief over intent) is not new. Prior
hierarchical BCIs mostly use scripted subroutines or user-taught skills for the autonomy layer. What this study adds is
(a) a CAVIN-style *learned* latent planner in that role, (b) a controlled sweep of decoder accuracy and vocabulary
size reporting success and user effort in bits, with an oracle executor that separates decoding from planning errors,
and (c) documented negative results. Because the scripted oracle beats the learned planner, (a) is exploratory: the
study does not show that learning the autonomy layer is advantageous. I did not find (in this limited search) a prior
work that reports the same sweep, but absence in a limited search is weak evidence and should not be stated as a
novelty claim.

## 4. Setup

**Environment** (`src/bcihp/env.py`). Planar tabletop with 3 zones and up to 3 discs plus up to 2 gray obstacles
placed on the straight path from target to zone in "blocked" scenes. A task succeeds when the target disc's centre is
within 3.5 cm of the zone centre and not held. Budget: 200 micro-steps. The oracle (`baselines.py`) clears blockers
by pushing and pushes the target, or weld-grasps and carries it over clutter; a grid A* planner avoids obstacles.

**Intent channel and protocol** (`commands.py`, `eval.py`). K possible tasks (object x zone). A decoded command is drawn from
a K x K confusion matrix (diagonal = decoder accuracy, 70% of errors to neighbouring indices). The system keeps a belief over
intents (Bayes rule with uniform prior) and **commits once the top belief reaches 0.9** or after 6 commands. Each decoded
command costs 10 idle micro-steps (decoding time) from the same budget. After committing, the user issues nothing further.
Commands per task is the effort metric; I also report an approximate information transfer (Wolpaw formula, which assumes
uniform errors and therefore only approximates my neighbour-biased noise).

**Baselines.** *Direct control*: five commands (left, up, right, down, grip toggle), 2 micro-steps per direction command,
same decoder accuracy, a simulated user who follows the oracle plan (optimistic). *Oracle*: same confirmation session,
scripted executor.

## 5. Method

**Model** (`model.py`, 0.5M parameters). State-conditioned priors p(c|s), p(z|s,c); meta-dynamics predicts the state after a
20-step macro step from (s, c); the action generator decodes (s, c, z) into 20 actions. Training maximises a cascaded ELBO with
Gaussian likelihoods, KL annealing, free bits, and weights beta_c = 0.5, beta_z = 0.1. Data: 50k (default) or 100k (mixed)
self-supervised windows from the oracle on random tasks (about 40%) and random push/carry/wander primitives (about 60%).

**Planner** (`planner.py`). Iteration 0 samples T = 3 effect codes sequentially from the prior; CEM refines the sequence;
the first effect is decoded with the *prior-mean* motion code; the first 5 actions run, then re-plan. The cost is the
distance of the target to the goal zone at the last predicted step (plus mean over steps) plus small shaping terms
(gripper-to-target distance, penalties for ending while holding).

**Differences from CAVIN** (so it is not called a reproduction): low-dimensional state instead of pixels; no
separate low-level dynamics model, so the two temporal resolutions are not both modelled; my motion-code posterior
conditions on the executed actions (my reading of the cascaded inference; verify against the paper); a hand-designed cost with
shaping; much smaller scale and a simpler environment.

## 6. Evaluation protocol

Seeds: pilot design on 5000+, ablations on 9000+, headline on 12000+ (fresh), mixed-scene investigations on 5000+
(development) and the mixed headline on 12000+. 100 episodes per cell; scenes paired across methods; 95% Wilson intervals
(`docs/results_table.md`). Re-planning every 5 steps was selected after the ablation seeds and re-validated on 12000+.

## 7. Results (default scene)

**7.1 Success and user input** (K = 9; unblocked / blocked scenes; commands and bits for unblocked scenes):

| accuracy | planner | oracle | direct | planner cmds (bits) | direct cmds (bits) |
|---|---|---|---|---|---|
| 0.5 | 0.67 / 0.63 | 0.87 / 0.87 | 0.15 / 0.24 | 5.0 (3.4) | 70.6 (22.7) |
| 0.6 | 0.74 / 0.70 | 0.93 / 0.93 | 0.34 / 0.44 | 4.3 (4.3) | 68.4 (37.7) |
| 0.7 | 0.79 / 0.76 | 0.99 / 0.99 | 0.63 / 0.64 | 3.1 (4.4) | 60.7 (51.0) |
| 0.8 | 0.76 / 0.74 | 0.96 / 0.96 | 0.84 / 0.84 | 2.4 (4.4) | 51.7 (62.0) |
| 0.9 | 0.74 / 0.73 | 0.94 / 0.94 | 0.95 / 0.95 | 1.0 (2.4) | 41.8 (69.1) |
| 1.0 | 0.80 / 0.79 | 1.00 / 1.00 | 1.00 / 1.00 | 1.0 (3.2) | 31.3 (72.6) |

At accuracy 0.8 the planner and direct control are statistically indistinguishable in success (overlapping intervals)
but differ by about 20x in commands.

**7.2 Vocabulary size.** At accuracy 1.0 the planner scores 0.73, 0.74, 0.80 (unblocked) for K = 3, 6, 9. At 0.5 it scores
0.47, 0.54, 0.67. The improvement with K is driven by how reliably the intent is confirmed (commit accuracy 0.71, 0.83, 0.87),
which depends on my noise model (neighbour-biased errors give more distinctive evidence at larger K). With real confusion
matrices the trend may differ; do not generalise it.

**7.3 Ablations** (accuracy 1.0, K = 9, seeds 9000+, correct goal; all use re-planning every 10 steps except the last row):

| variant | unblocked | blocked |
|---|---|---|
| effect horizon 1 | 0.21 (0.14-0.30) | 0.12 (0.07-0.20) |
| effect horizon 2 | 0.62 (0.52-0.71) | 0.52 (0.42-0.62) |
| **effect horizon 3 (default)** | 0.77 (0.68-0.84) | 0.70 (0.60-0.78) |
| effect horizon 4 | 0.78 (0.69-0.85) | 0.72 (0.63-0.80) |
| no cost shaping | 0.81 (0.72-0.87) | 0.73 (0.64-0.81) |
| prior sampling only (no CEM) | 0.72 (0.63-0.80) | 0.73 (0.64-0.81) |
| replan every 5 steps | 0.87 (0.79-0.92) | 0.80 (0.71-0.87) |

Findings: planning depth matters and saturates at 3 steps. Cost shaping and CEM refinement show **no demonstrated benefit**
(an apparent shaping gain on pilot seeds did not replicate on fresh seeds). Replanning every 5 steps helps; on the headline seeds
the planner scores 0.80 / 0.79 at accuracy 1.0, lower than the 0.87 / 0.80 on the selection seeds, as expected after selection.

## 8. Results: mixed-skill scene and the pushing failure

Five discs (three light = graspable, two heavy = push only), 3 zones, K = 15 tasks, 100k training windows, seeds 12000+.
Overall success (unblocked / blocked): planner 0.42 / 0.37 at accuracy 1.0, oracle 1.00 / 0.91, direct 0.47 / 0.46.
Splitting by target type (accuracy 1.0; n = 45 light, 55 heavy tasks per scene type):

| method | light (grasp) unblocked / blocked | heavy (push) unblocked / blocked |
|---|---|---|
| planner | 0.87 / 0.78 | 0.05 / 0.04 |
| oracle | 1.00 / 1.00 | 1.00 / 0.84 |
| direct | 1.00 / 1.00 | 0.04 / 0.02 |

The scene is solvable (the oracle pushes), so the failure belongs to the learned planner. With the correct goal and no decoding
noise, heavy-target failures are mostly "missed zone" (56-61%, median final distance 34-49 cm) and "disturbed other discs"
(28-39%), not "never moved" (0-11%): the planner moves discs, but in the wrong directions. For pushes the model predicts about
11 cm of target progress per chunk and the actions deliver about 3.5 cm; for grasps about 13 cm predicted vs 7-8 cm delivered.
Direct control also fails on heavy targets because four-direction commands cannot push a round disc accurately along a diagonal;
this is a property of the baseline, not evidence that pushing is impossible by BCI.

**Attempted fixes (development seeds 5000+, accuracy 1.0; light n = 66, heavy n = 34 per scene type):**

| attempt | light unblocked / blocked | heavy unblocked / blocked |
|---|---|---|
| baseline | 0.89 / 0.73 | 0.03 / 0.00 |
| remove the gripper-to-centre cost term for heavy targets | 0.89 / 0.73 | 0.06 / 0.00 |
| + push-geometry cost (gripper behind the disc), weight 0.3 | 0.89 / 0.73 | 0.00 / 0.03 |
| + push-geometry cost, weight 1.0 | 0.89 / 0.73 | 0.00 / 0.00 |
| retrain with beta_z = 1.0 (information moved from z to c) | 0.68 / 0.68 | 0.03 / 0.00 |
| retrain with beta_c = 0.2, beta_z = 1.0 | 0.41 / 0.35 | 0.00 / 0.00 |

None cleared a pre-set bar (heavy success of at least 0.25 with light success maintained). A latent-structure check on
the saved model showed that replacing the posterior motion code by its prior mean (as the planner does) loses about
61% of the action information on push windows, 44% on grasp-carry windows and 42% on windows without object motion, so
a loss of motion information is plausible but does not by itself explain the push failure, since grasping tolerates a
similar loss. A push-only world without heavy objects gave the same pattern (planner 0.00 / 0.01 vs oracle 1.00 / 0.96,
100 episodes, seeds 9000+). Untested remaining hypotheses: on-policy data aggregation focused on the planner's own
push failures; a meta-dynamics model that also depends on z; a low-level dynamics model as in the original CAVIN.

## 9. Limitations

* Simulated decoder and user; no EEG, no latency, no non-stationarity, no fatigue; commands are independent draws.
* Abstract manipulation (planar discs, weld grasp, cosmetic arm, state observations).
* Planner reliability (0.63-0.80) is below the oracle and far below anything usable for deployment.
* The scripted oracle outperforms the learned planner; the value of *learning* the autonomy layer is not shown.
* Pushing is unsolved by the learned planner; the direct baseline also cannot push.
* Cost shaping is hand-designed; its benefit is unproven; hyperparameters were chosen on pilot or ablation seeds
  (mitigated by fresh headline seeds, but many comparisons were made).
* The vocabulary-size trend depends on my noise model; the Wolpaw bit estimate assumes uniform errors.
* No user study; nothing here measures perceived control, trust or workload.
* CAVIN-inspired, not a reproduction.

## 10. What would change with real decoded signals, and next steps

Replace the synthetic confusion matrix with matrices measured from real multi-class motor-imagery decoders (per subject,
including poorly performing users); expect non-uniform errors, drift across sessions, and a time cost per decision.
Add an error-related-potential channel so the user can veto a wrong commit (committed-wrong rates here reach 11-36% at
accuracy 0.5-0.6). Show the planner's predicted subgoal to the user before execution ("predictive transparency"), which
connects to explainable-AI work on biosignal systems. Improve the learned planner with on-policy data aggregation. Then,
and only then, a user study.

## 11. Reproducibility

Commands are in the README. Scripts: `03_train.py`, `04_evaluate.py`, `09_breakdown.py` (per-skill split from the evaluation
CSV), `10_latent_check.py`. Seeds, protocol and every reported number are in `results/*.csv` and `docs/results_table*.md`.

## References (verify details before submission)

* Fang, Zhu, Garg, Savarese, Fei-Fei. Dynamics Learning with Cascaded Variational Inference for Multi-Step Manipulation. CoRL 2019. Code: github.com/StanfordVL/cavin
* Xu, Nair, Zhu, Gao, Garg, Fei-Fei, Savarese. Neural Task Programming: Learning to Generalize Across Hierarchical Tasks. ICRA 2018.
* Huang, Nair, Xu, Zhu, Garg, Fei-Fei, Savarese, Niebles. Neural Task Graphs: Generalizing to Unseen Tasks from a Single Video Demonstration. CVPR 2019.
* Losey et al. Controlling Assistive Robots with Learned Latent Actions (ICRA 2020; arXiv 1909.09674) and Learning Latent Actions to Control Assistive Robots (arXiv 2107.02907).
* Javdani, Srinivasa, Bagnell. Shared Autonomy via Hindsight Optimization. RSS 2015.
* Muelling, Venkatraman, Valois, Downey, Weiss, Javdani, Hebert, Schwartz, Collinger, Bagnell. Autonomy Infused Teleoperation with Application to BCI Manipulation. RSS 2015 (arXiv 1503.05451) [verify author list and venue].
* Iturrate, Montesano, Minguez. Shared-control brain-computer interface for a two dimensional reaching task using EEG error-related potentials. EMBC 2013. Iturrate, Montesano, Minguez. Single trial recognition of error-related potentials during observation of robot operation. EMBC 2010.
* Chung, Cheung, Scherer, Rao. A Hierarchical Architecture for Adaptive Brain-Computer Interfacing. IJCAI 2011.
* Task Level Hierarchical System for BCI-enabled Shared Autonomy. IEEE-RAS Humanoids 2017 (Columbia University) [verify authors].
* Burget et al. Acting thoughts: towards a mobile robotic service assistant for users with limited communication skills. ECMR 2017 [verify title and authors].
* Shared control of a robotic arm using non-invasive brain-computer interface and computer vision guidance. Robotics and Autonomous Systems 115 (2019) [verify authors].
* Douglas et al. Levels of shared autonomy in brain-robot interfaces: enabling multi-robot multi-human collaboration for activities of daily living [verify venue and year].
* Human-Centered Shared Autonomy for Motor Planning, Learning, and Control Applications. arXiv 2506.16044 (review).
* ESANN 2009, paper "es2009-115" on augmenting information from BCIs for assistive devices [verify title and authors].
