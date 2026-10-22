# Number macros for the paper (auto-generated: `python -m analysis.paper_numbers_v2` in the code repo)

Type the macro where the number goes, e.g. `a gain of \gainMean points [\gainLo, \gainHi]`. Signs: negative values carry
their own minus; write `+\gainMean` yourself when you want a plus. `CHECK` compares with the value in CLAIMS_LIST.md.

**Rounding tie in the headline.** The exact values are BC 54.55%, imagination 81.70%, gain 27.15 points. To one decimal they
round to 54.6, 81.7 and 27.2, and 81.7 - 54.6 reads as 27.1. Wherever the two catch rates and the gain appear together
(the main results sentence, the main table), use `\bcCatchExact`, `\imagCatchExact`, `\gainMeanExact` [`\gainLoExact`, `\gainHiExact`].

| macro | value | meaning | claim | source | CHECK |
|---|---|---|---|---|---|
| `\nDraws` | 8 | Phase-2 retrains (draws) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\nSeeds` | 3 | imagination seeds per draw | C1 | phase3/analysis/p3_runs.csv | ok |
| `\nRuns` | 24 | imagination runs | C1 | phase3/analysis/p3_runs.csv | ok |
| `\nBoards` | 500 | evaluation games per policy | C1 | evaluate_env --num-episodes | ok |
| `\gainMean` | 27.2 | mean catch-rate gain over the BC parent, points (mean of draw means) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainLo` | 22.2 | 95% CI lower (t, df 7) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainHi` | 32.1 | 95% CI upper | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainT` | 12.98 | t statistic, df 7 | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainMeanExact` | 27.15 | the same mean gain to two decimals (use where the two catch rates are also shown) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainLoExact` | 22.20 | 95% CI lower, two decimals | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainHiExact` | 32.10 | 95% CI upper, two decimals | C1 | phase3/analysis/p3_runs.csv | ok |
| `\nRunsPositive` | 24 | runs with a positive gain | C1 | phase3/analysis/p3_runs.csv | ok |
| `\nDrawsPositive` | 8 | draw means with a positive gain | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainRunMin` | 13.0 | smallest single-run gain, points | C1 | phase3/analysis/p3_runs.csv | ok |
| `\gainRunMax` | 36.2 | largest single-run gain, points | C1 | phase3/analysis/p3_runs.csv |  |
| `\runSEmin` | 2.5 | per-run paired standard error, smallest (points) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\runSEmax` | 3.0 | per-run paired standard error, largest | C1 | phase3/analysis/p3_runs.csv | ok |
| `\bcCatch` | 54.6 | BC parents' mean catch rate, % (8 draws); exactly 54.55, a rounding tie | C1 | phase3/analysis/p3_runs.csv | ok |
| `\bcCatchExact` | 54.55 | BC parents' mean catch rate to two decimals | C1 | phase3/analysis/p3_runs.csv | ok |
| `\imagCatchExact` | 81.70 | imagination policies' mean catch rate to two decimals | C1 | phase3/analysis/p3_runs.csv | ok |
| `\bcCatchMin` | 51.2 | lowest BC parent catch rate, % | C1 | phase3/analysis/p3_runs.csv |  |
| `\bcCatchMax` | 56.6 | highest BC parent catch rate, % | C1 | phase3/analysis/p3_runs.csv |  |
| `\imagCatch` | 81.7 | imagination policies' mean catch rate, % (24 runs) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\imagCatchMin` | 68.8 | lowest imagination-run catch rate, % | C1 | phase3/analysis/p3_runs.csv | ok |
| `\imagCatchMax` | 90.6 | highest imagination-run catch rate, % | C1 | phase3/analysis/p3_runs.csv | ok |
| `\bcReturnCaught` | 536 | BC parents: mean return of caught games (time the ball was held) | C1 | phase3/analysis/p3_runs.csv | ok |
| `\imagReturnCaught` | 637 | imagination policies: mean return of caught games | C1 | phase3/analysis/p3_runs.csv | ok |
| `\bcReturn` | 292 | BC parents' mean return | C1 | phase3/analysis/p3_runs.csv | ok |
| `\imagReturn` | 521 | imagination policies' mean return | C1 | phase3/analysis/p3_runs.csv | ok |
| `\retBothBC` | 549 | BC parents: mean return on the games both the run and its parent caught (mean over 24 runs) | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retBothImag` | 665 | imagination runs: mean return on the same games | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retBothGain` | 116 | paired return gain on those games, mean of the 8 draw means | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retBothLo` | 87 | its 95% t interval over the 8 draw means, lower | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retBothHi` | 146 | ... upper | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retBothRunsPos` | 24 | runs whose paired return gain on those games is positive (of 24) | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\retExtraCatch` | 603 | imagination runs: mean return on the games only the run caught (its parent missed) | C1 | evaluation/tmlr-aligned-bc-seed*-n500 + phase3/draw*/evaluation/tmlr-aligned-p3-draw*-seed*-n500 episodes.csv | ok |
| `\sdBetween` | 5.70 | between-draw sd of the gain, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdBetweenLo` | 3.43 | Williams-Tukey 95% lower | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdBetweenHi` | 11.95 | Williams-Tukey 95% upper | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdWithin` | 2.76 | within-draw (seed-to-seed) sd, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdWithinLo` | 2.05 | 95% lower | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdWithinHi` | 4.19 | 95% upper | C2 | phase3/analysis/p3_runs.csv | ok |
| `\icc` | 0.81 | intra-class correlation (raw; keep out of the abstract) | C2 | phase3/analysis/p3_runs.csv | ok |
| `\drawMeanMin` | 15.2 | smallest draw-mean gain, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\drawMeanMax` | 33.5 | largest draw-mean gain, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looLowDraw` | 10 | the draw with the smallest mean gain | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looWithoutLow` | 3.33 | between-draw sd with that draw left out | C2 | phase3/analysis/p3_runs.csv | ok |
| `\parentSdPts` | 1.77 | observed sd of the 8 BC parent catch rates, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdBetweenCorr` | 5.4 | between-draw sd after subtracting the parents' observed variance (secondary) | C2 | phase3/analysis/p3_runs.csv | ok |
| `\parentBinomSdPts` | 2.23 | binomial sd of one parent's 500 game catch rate at the observed rates, points | C2 | phase3/analysis/p3_runs.csv | ok |
| `\sdBetweenCorrBinom` | 5.2 | between-draw sd after subtracting the parents' binomial variance (secondary) | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looWithoutLowCorrBinom` | 2.5 | ... with the lowest draw left out (secondary) | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looWithoutLowCorr` | 2.8 | ... with the lowest draw left out (secondary) | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looOtherMin` | 5.50 | between-draw sd leaving out any other draw, smallest | C2 | phase3/analysis/p3_runs.csv | ok |
| `\looOtherMax` | 6.17 | ... largest | C2 | phase3/analysis/p3_runs.csv | ok |
| `\binomShare` | 39 | share of within-draw variance that is binomial evaluation noise, % | C2 | phase3/analysis/p3_runs.csv | ok |
| `\stopDiff` | -0.15 | catch rate at epoch 15 minus epoch 10, points (8 draws, seed 1) | C3 | phase3/draw*/evaluation/*-seed1[-e10]-n500/summary.json | ok |
| `\stopLo` | -1.94 | 95% CI lower | C3 | phase3/draw*/evaluation/*-seed1[-e10]-n500/summary.json | ok |
| `\stopHalfWidth` | 1.79 | half width of the epoch 10 vs 15 paired t interval, points | C3 | phase3/draw*/evaluation/*-seed1[-e10]-n500/summary.json | ok |
| `\stopHi` | 1.64 | 95% CI upper | C3 | phase3/draw*/evaluation/*-seed1[-e10]-n500/summary.json | ok |
| `\riseWindow` | 500 | training steps in each window of the imagined return comparison | C3 | phase3/analysis/p3_wandb_curves.csv (last500 vs prev500 columns) | ok |
| `\bcNllDropPct` | 0.07 | held out BC loss change from epoch 32 to 40 as % of its value (Phase 2 stop check) | C3 | phase2/eval_stop_criterion/summary.json | ok |
| `\nStopDraws` | 5 | Phase 2 draws whose checkpoints were scored in that check | C3 | phase2/eval_stop_criterion/summary.json | ok |
| `\bcLossEpochFrom` | 32 | earlier Phase 2 checkpoint in the held out BC loss comparison (checkpoints at epochs 8, 16, 24, 32, 40) | C3 | CONSTANT: phase2/eval_stop_criterion/summary.json (bc0_clean_delta_32_40) | ok |
| `\bcNllTrainOverVal` | 1.07 | ratio of the BC head's training loss to its held out loss at epoch 40 (above 1 means no memorization) | C3 | phase2/eval_stop_criterion/summary.json | ok |
| `\imagRisePct` | 3.0 | rise of imagined return over the last 500 steps, % | C3 | phase3/analysis/p3_wandb_curves.csv | ok |
| `\imagRiseRuns` | 23 | runs whose imagined return was still rising | C3 | phase3/analysis/p3_wandb_curves.csv | ok |
| `\oldAnchorGain` | 5.9 | OLD pipeline, anchor draw: mean gain over its BC parent, points (6 runs) | C4 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\oldAnchorLo` | 1.5 | 95% CI lower (t, df 5) | C4 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\oldAnchorHi` | 10.4 | 95% CI upper | C4 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\oldAnchorRuns` | 6 | anchor imagination runs | C4 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\oldAnchorPositive` | 6 | anchor runs with a positive gain | C4 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\oldRetrainWorst` | -32.6 | OLD pipeline: mean gain on the collapsing retrain (seed 11), points | C4 | evaluation/tmlr-bc-seed*, tmlr-fact-* | ok |
| `\oldRetrainBest` | 24.2 | OLD pipeline: mean gain on retrain seed 12 | C4 | evaluation/tmlr-bc-seed*, tmlr-fact-* | ok |
| `\oldRetrainThird` | -8.3 | OLD pipeline: mean gain on retrain seed 13 | C4 | evaluation/tmlr-bc-seed*, tmlr-fact-* | ok |
| `\oldBcCatchMin` | 35.6 | OLD BC parents: lowest catch rate, % | C4 | evaluation/tmlr-bc-seed*, tmlr-fact-* | ok |
| `\oldBcCatchMax` | 44.8 | OLD BC parents: highest catch rate, % | C4 | evaluation/tmlr-bc-seed*, tmlr-fact-* | ok |
| `\randomFloor` | 9.4 | catch rate of a random policy, % | C12 | evaluation/tmlr-n500-categorical, tmlr-cat-seed* | ok |
| `\rewMaeNewMin` | 0.038 | reward-model error per step, new imagination runs, smallest | C5 | summary.json reward_calibration | ok |
| `\rewMaeNewMax` | 0.061 | ... largest | C5 | summary.json reward_calibration | ok |
| `\rewCorrNewMin` | 0.972 | reward-model correlation, new runs, smallest | C5 | summary.json reward_calibration | ok |
| `\rewCorrNewMax` | 0.985 | ... largest | C5 | summary.json reward_calibration | ok |
| `\rewMaeParMin` | 0.029 | reward-model error, new BC parents, smallest | C5 | summary.json reward_calibration | ok |
| `\rewMaeParMax` | 0.031 | ... largest | C5 | summary.json reward_calibration | ok |
| `\rewMaeOldMin` | 0.17 | reward-model error, OLD collapsing draw, smaller of 2 runs | C5 | summary.json reward_calibration | ok |
| `\rewMaeOldMax` | 0.26 | ... larger | C5 | summary.json reward_calibration | ok |
| `\rewCorrOldMin` | 0.61 | reward-model correlation, OLD collapsing draw, smaller | C5 | summary.json reward_calibration | ok |
| `\rewCorrOldMax` | 0.71 | ... larger | C5 | summary.json reward_calibration | ok |
| `\mirageNew` | 0 | mirage games (never caught, believed > 100), new runs | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\missedNew` | 437 | never-caught games, new runs | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\mirageOld` | 91 | mirage games, OLD collapsing draw | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\missedOld` | 173 | never-caught games, OLD collapsing draw | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\mirageOldHealthy` | 3 | mirage games, OLD healthy runs | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\missedOldHealthy` | 530 | never-caught games, OLD healthy runs | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\mirageOldParent` | 0 | mirage games of the collapsed retrain's BC parent, same reward head, same first games | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\missedOldParent` | 54 | never-caught games of that parent | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\rewCorrOldParent` | 0.98 | reward correlation of that parent's head on its own 500 games | C5 | evaluation/tmlr-bc-seed11-n500/summary.json | ok |
| `\maxBeliefMissedNew` | 46 | highest predicted return on a never-caught game, new runs | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\maxBeliefMissedOld` | 978 | ... OLD collapsing draw | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) |  |
| `\mirageThreshold` | 100 | predicted return above which a never-caught game counts as a mirage | C5 | phase3/analysis/scripts/perstep_metrics.py THRESH | ok |
| `\nPerstepBoards` | 100 | boards per run in the per-game reward analysis (boards 0-99) | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\phantomChildMed` | 0.030 | phantom reward per zero-reward step, new children, median | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\phantomParentMed` | 0.011 | ... new BC parents, median | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\phantomOldMin` | 0.196 | ... OLD collapsing draw, smaller | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\phantomOldMax` | 0.291 | ... larger | C5 | phase3/analysis/perstep_metrics.csv (boards 0-99) | ok |
| `\timingNewDrawsAmin` | 64 | 8 new Phase-2 draws, training-style setup: late-action penalty, smallest, % | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewDrawsAmax` | 66 | ... largest | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewDrawsBmin` | 177 | 8 new draws, pure-noise setup: late-action penalty, smallest, % | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewDrawsBmax` | 181 | ... largest | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewDrawsZmin` | 10.6 | 8 new Phase-2 draws, both setups: episode-clustered z of late vs correct, smallest | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewDrawsCorrect` | 16 | new-draw setups (of 16) with lowest error at correct timing | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingOldReleasedA` | -1.8 | OLD released Phase-1 (epoch 320), training-style: change with late actions, % | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingOldReleasedB` | -20.7 | ... pure-noise setup, % | C6 | timing_new_draws/timing_summary.csv | ok |
| `\timingNewPoneA` | 53 | corrected Phase-1 (epoch 320), training-style: late-action penalty, % | C6 | eval_held_out/final_checks.out | ok |
| `\timingNewPoneB` | 121 | ... pure-noise setup, % | C6 | eval_held_out/final_checks.out | ok |
| `\timingOldDrawsMin` | -3.3 | 4 OLD Phase-2 draws, both setups: change with late actions, most negative, % | C6 | old_draw_forensics/timing_tests.log | ok |
| `\timingOldDrawsMax` | -1.0 | ... least negative, % | C6 | old_draw_forensics/timing_tests.log | ok |
| `\timingOldDrawsZnear` | -3.0 | 4 OLD Phase-2 draws, both setups: episode-clustered z of late vs correct, closest to zero | C6 | old_draw_forensics/timing_tests.log | ok |
| `\timingOldDrawsZfar` | -8.6 | ... farthest from zero | C6 | old_draw_forensics/timing_tests.log | ok |
| `\timingOldDrawsLate` | 8 | old-draw setups (of 8) with lowest error at LATE timing | C6 | old_draw_forensics/timing_tests.log | ok |
| `\shuffleOldDrawsAmin` | 22 | 4 OLD Phase-2 draws, training-style: error increase with shuffled vs correctly timed actions, smallest, % | C8 | old_draw_forensics/timing_tests.log | ok |
| `\shuffleOldDrawsAmax` | 52 | ... largest, % | C8 | old_draw_forensics/timing_tests.log | ok |
| `\shuffleOldDrawsBmin` | 52 | 4 OLD Phase-2 draws, pure-noise setup: error increase with shuffled actions, smallest, % | C8 | old_draw_forensics/timing_tests.log | ok |
| `\shuffleOldDrawsBmax` | 92 | ... largest, % | C8 | old_draw_forensics/timing_tests.log | ok |
| `\shufPolicyShiftMin` | 0.14 | checklist shuffle probe: policy head shift per action dimension, old Phase 2 models (categorical and Gaussian), smallest | C8 | evaluation/agent{-e40-categorical,-optionA-e040,}/action_shuffle_agent.csv | ok |
| `\shufPolicyShiftMax` | 0.19 | ... largest | C8 | evaluation/agent{-e40-categorical,-optionA-e040,}/action_shuffle_agent.csv | ok |
| `\shufPolicyShiftBlindMin` | 0.005 | same probe on the action blind model (agent embedding frozen at init), smallest | C8 | evaluation/agent{-e40-categorical,-optionA-e040,}/action_shuffle_agent.csv | ok |
| `\shufPolicyShiftBlindMax` | 0.006 | ... largest | C8 | evaluation/agent{-e40-categorical,-optionA-e040,}/action_shuffle_agent.csv | ok |
| `\timingOldEarlyA` | -11.5 | OLD Phase-1 epoch 40, training-style: change with late actions, % | C6 | eval_held_out/timing_control_OLD_e040.log | ok |
| `\nHeldOut` | 24 | held-out episodes behind every world-model test | C6 | OfflineDataset split=val | ok |
| `\errOldSmallStep` | 0.0069 | OLD released model: 1-step prediction error at the smallest step (1/64) | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errOldShortcutMin` | 0.52 | OLD released model: error at the six shortcut steps, smallest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errOldShortcutMax` | 0.75 | ... largest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errOldShortcutRatioMin` | 76 | shortcut-step error as a multiple of the smallest-step error, smallest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errOldShortcutRatioMax` | 109 | ... largest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errNewMin` | 0.0058 | corrected model: error over all seven step sizes, smallest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errNewMax` | 0.0215 | ... largest | C7 | eval_held_out/evaldyn_*/d_bucket_metrics.csv | ok |
| `\errOldReleased` | 0.5261 | OLD released (epoch 320): held-out 1-step error, all step sizes | C7 | evaldyn_OLD_e320/summary.json | ok |
| `\errOldEarly` | 0.0143 | OLD epoch 40: held-out 1-step error | C7 | evaldyn_OLD_e040/summary.json | ok |
| `\errNew` | 0.0096 | corrected (epoch 320): held-out 1-step error | C7 | evaldyn_JOINT_e320/summary.json | ok |
| `\tdOldShortMin` | 0.49 | OLD released: error at signal level 0.6-0.8 (no frame-0 entries), shortcut steps, smallest | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdOldShortMax` | 0.57 | ... largest | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdOldSmall` | 0.0037 | ... at the smallest step 1/64 | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdNewMin` | 0.0039 | corrected: error at signal level 0.6-0.8, all step sizes present, smallest | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdNewMax` | 0.0049 | ... largest | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\errOldNoTopMin` | 13 | OLD released, signal levels below 0.8 (where frame 0 never sits): shortcut-step error as a multiple of the smallest-step error, five steps below 1, smallest | C7 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\errOldNoTopMax` | 24 | ... largest | C7 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdTauLo` | 0.6 | lower edge of the signal-level bin used for the td* macros | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\tdTauHi` | 0.8 | upper edge of that bin | C9 | eval_held_out/evaldyn_{OLD,JOINT}_e320/tau_d_joint_metrics.csv | ok |
| `\nTimingModels` | 9 | models in the 2026-09-29 timing run (8 new Phase 2 draws + old Phase 1 epoch 320) | C9 | timing_new_draws/timing_summary.csv | ok |
| `\timingMinutes` | 14 | wall time of that run on a laptop GPU, minutes | C9 | CONSTANT: sheet CA, run 03:23:03-03:36:50Z | ok |
| `\errOldOverNew` | 55 | OLD released error as a multiple of the corrected error (same evaluator, same held-out episodes): the absolute anchor the old train-vs-held-out check lacked | C8 | evaldyn_OLD_e320 + evaldyn_JOINT_e320 summary.json | ok |
| `\rollOldReleased` | 0.5678 | OLD released: 4-step rollout error | C7 | evaldyn_OLD_e320/summary.json | ok |
| `\rollPairedOld` | 0.0303 | paired 4-step rollout error, OLD epoch 40 (1,280 shared windows) | C7 | eval_held_out/final_checks.out | ok |
| `\rollPairedNew` | 0.0189 | paired 4-step rollout error, corrected epoch 320 | C7 | eval_held_out/final_checks.out | ok |
| `\rollPairedPct` | -37.6 | paired rollout change, corrected vs OLD epoch 40, % | C7 | eval_held_out/final_checks.out | ok |
| `\rollPairedZ` | -12.0 | its episode-clustered z | C7 | eval_held_out/final_checks.out | ok |
| `\shuffleOld` | 166 | OLD epoch 40: error increase with shuffled actions, % (the shuffle probe passes) | C8 | evaldyn_OLD_e040/action_shuffle_sensitivity.csv | ok |
| `\wmTrain` | 0.531 | OLD released model: world-model error on training episodes | C8 | evaluation/wm-split-check | ok |
| `\wmHeldOut` | 0.528 | ... on held-out episodes (the 'no gap' check) | C8 | evaluation/wm-split-check | ok |
| `\wmRatio` | 37 | held-out error as a multiple of the same run's epoch-40 error (latent MSE) | C8 | derived | ok |
| `\nChecks` | 12 | checks in the v1 verification matrix | C8 | old/main_v1.tex Table 1 | ok |
| `\chkTokRank` | 11/16 | v1 tokenizer check: latent rank over 16 dims | C8 | old/main_v1.tex at tag v1-old-pipeline, Table 1 (lines 442-447) [CONSTANT] | ok |
| `\chkRewMaeMin` | 0.013 | v1 reward head closed loop calibration MAE, smallest | C8 | old/main_v1.tex at tag v1-old-pipeline, Table 1 (lines 442-447) [CONSTANT] | ok |
| `\chkRewMaeMax` | 0.035 | v1 reward head closed loop calibration MAE, largest | C8 | old/main_v1.tex at tag v1-old-pipeline, Table 1 (lines 442-447) [CONSTANT] | ok |
| `\chkContBrier` | 0.0020 | v1 continue head closed loop Brier score | C8 | old/main_v1.tex at tag v1-old-pipeline, Table 1 (lines 442-447) [CONSTANT] | ok |
| `\chkReplayN` | 90 | v1 determinism check: replayed outcomes identical, out of this many | C8 | old/main_v1.tex at tag v1-old-pipeline, Table 1 (lines 442-447) [CONSTANT] | ok |
| `\rhoAll` | 0.14 | Spearman, imagined return vs real catch, 24 runs | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\rhoLo` | -0.60 | draw-then-seed bootstrap 95% lower | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\rhoHi` | 0.88 | ... upper | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\rhoDraws` | 0.26 | Spearman over the 8 draw means | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\rhoDrawsP` | 0.54 | its exact permutation p | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\pickTopCatch` | 85.4 | real catch of the run with the highest imagined return, % | C10 | p3_runs.csv + p3_wandb_curves.csv (as phase3/analysis/scripts/imagined_vs_real.py) | ok |
| `\detMedian` | 18.3 | median deterministic catch over 32 evaluations, % | C12 | det_readout_check/readout_table.csv | ok |
| `\sampledMedian` | 62.7 | median sampled catch over the same 16 policies, % | C12 | det_readout_check/readout_table.csv | ok |
| `\bcArgmax` | 11.6 | BC parents, argmax readout, median catch % | C12 | det_readout_check/readout_table.csv | ok |
| `\bcMeanReadout` | 11.3 | BC parents, mean readout, median catch % | C12 | det_readout_check/readout_table.csv | ok |
| `\imagArgmax` | 26.5 | imagination children (seed 1), argmax readout, median catch % | C12 | det_readout_check/readout_table.csv | ok |
| `\imagMeanReadout` | 31.6 | imagination children, mean readout, median catch % | C12 | det_readout_check/readout_table.csv | ok |
| `\imagSampledSeedOne` | 84.2 | the same 8 children with sampled actions, median catch % | C12 | det_readout_check/readout_table.csv | ok |
| `\gpuDivergedBoards` | 11 | boards (of 500) whose trajectory diverged between the two GPU families on the one re-evaluated run | C14 | CONSTANT: sheet BU cross-GPU check (draw 13 seed 1 evaluated on a 4090 and an A4500) | ok |
| `\gpuCatchFlips` | 1 | catch outcomes that flipped on that run (453 vs 452 catches) | C14 | CONSTANT: sheet BU cross-GPU check (draw 13 seed 1 evaluated on a 4090 and an A4500) | ok |
| `\gpuWorstCase` | 2.2 | worst case if every diverged board flipped one way, points | C14 | CONSTANT: sheet BU cross-GPU check (draw 13 seed 1 evaluated on a 4090 and an A4500) | ok |
| `\nAmpereRuns` | 12 | imagination policies (draws 10, 13, 14, 17) evaluated on the Ampere family; all BC parents on one Blackwell card (RTX 5070 Ti) | C14 | CONSTANT: sheet BU cross-GPU check (draw 13 seed 1 evaluated on a 4090 and an A4500) | ok |
| `\nLaunchesPThree` | 27 | wandb launches behind the 24 imagination results (3 aborted and relaunched with the same seeds) | C14 | CONSTANT: sheet CC | ok |
| `\nAbortedPThree` | 3 | aborted imagination launches (one killed on a slow pod, two failed on a corrupted tokenizer file) | C14 | CONSTANT: sheet CC | ok |
| `\nLaunchesPOne` | 3 | Phase 1 launches on the aligned data (collapsed, stopped at its banner, corrected) | C14 | backup phase1 logs: phase1_train_COLLAPSED_run1.log, phase1_joint_train_OFFLINE_ATTEMPT_aborted.log, phase1_joint_train.log | ok |
| `\nResultsPOne` | 1 | Phase 1 models used by every later phase | C14 | CONSTANT: SETUP_FACTS Phase 1 | ok |
| `\nAbortedPTwo` | 2 | aborted Phase 2 launches (one stopped to restart its pod at epoch 8, one out of GPU memory at start) | C14 | CONSTANT: SETUP_FACTS launch log; sheet CC | ok |
| `\nLaunchesPTwo` | 10 | wandb launches behind the 8 Phase 2 draws (2 aborted) | C14 | CONSTANT: sheet CC | ok |
| `\actionRepeat` | 2 | action repeat in the simulator | setup | evaluation/tmlr-aligned-bc-seed10-n500/summary.json (same for every evaluation) | ok |
| `\simStepsPerGame` | 1000 | simulator steps per game (decisions x action repeat) | setup | evaluation/tmlr-aligned-bc-seed10-n500/summary.json (same for every evaluation) | ok |
| `\returnMax` | 1000 | largest possible return (reward 1 per simulator step in the cup) | setup | evaluation/tmlr-aligned-bc-seed10-n500/summary.json (same for every evaluation) | ok |
| `\mujocoVersion` | 3.3.7 | MuJoCo version recorded by the evaluator | setup | evaluation/tmlr-aligned-bc-seed10-n500/summary.json (same for every evaluation) | ok |
| `\bootReplicates` | 20000 | replicates of the draw-then-seed bootstrap | C10 | phase3/analysis/scripts/imagined_vs_real.py | ok |
| `\nReadoutEvals` | 32 | deterministic evaluations (16 policies x 2 readouts) | C12 | det_readout_check/readout_table.csv | ok |
| `\demoReturn` | 574 | demonstrations: mean return (unpaired reference) | C1 | ball_in_cup_catch.npz rewards | ok |
| `\demoReturnCaught` | 682 | demonstrations: mean return of the caught episodes (time the ball was held) | C1 | ball_in_cup_catch.npz rewards | ok |
| `\demoCatch` | 84.2 | share of the demonstration episodes that catch, % (unpaired) | C1 | ball_in_cup_catch.npz rewards | ok |
| `\nDemos` | 240 | demonstration episodes | C1 | ball_in_cup_catch.npz | ok |
| `\lateCorrMin` | 0.47 | correlation of each stored action with the next one in its episode (a one-step-late action vs the correct one), smaller of the two action dimensions | C8 | ball_in_cup_catch.npz actions, rows 1-500 | ok |
| `\lateCorrMax` | 0.49 | ... larger of the two | C8 | ball_in_cup_catch.npz actions, rows 1-500 | ok |
| `\nRowZero` | 240 | episodes whose action row 0 is exactly zero in the original file (model-free tell of the 'led to' convention) | C9 | ball_in_cup_catch.npz actions, row 0 | ok |
| `\nParamsTokenizer` | 74.9 | tokenizer parameters, millions | setup | checkpoints: tokenizer_epoch_500.pt, phase2 dynamics_epoch_040.pt (state dict sizes) | ok |
| `\nParamsWorldModel` | 47.8 | world model (dynamics transformer) parameters, millions | setup | checkpoints: tokenizer_epoch_500.pt, phase2 dynamics_epoch_040.pt (state dict sizes) | ok |
| `\nParamsHeads` | 3.9 | reward, continue and policy heads, millions | setup | checkpoints: tokenizer_epoch_500.pt, phase2 dynamics_epoch_040.pt (state dict sizes) | ok |
| `\dreamerParamsTokenizer` | 400 | Dreamer 4 tokenizer parameters, millions (paper text) | setup | CONSTANT: arXiv 2509.24527 sec. 4 | ok |
| `\dreamerParamsDynamics` | 1.6 | Dreamer 4 dynamics model parameters, billions (paper text) | setup | CONSTANT: arXiv 2509.24527 sec. 4 | ok |
| `\tokPatch` | 16 | tokenizer patch size, pixels | setup | tokenizer_cfg | ok |
| `\tokDepth` | 8 | tokenizer transformer depth | setup | tokenizer_cfg | ok |
| `\embedDim` | 512 | embedding width (tokenizer and world model) | setup | dynamics_cfg | ok |
| `\latentDim` | 16 | latent dimension per latent token | setup | tokenizer_cfg | ok |
| `\latentTokens` | 32 | latent tokens per frame | setup | tokenizer_cfg | ok |
| `\dynDepth` | 12 | world model transformer depth | setup | dynamics_cfg | ok |
| `\dynHeads` | 8 | attention heads | setup | dynamics_cfg | ok |
| `\dynKvHeads` | 2 | key/value heads (grouped query attention) | setup | dynamics_cfg | ok |
| `\lrTok` | 1\times10^{-4} | tokenizer learning rate | setup | tokenizer_cfg | ok |
| `\wdTok` | 0.05 | tokenizer weight decay | setup | tokenizer_cfg | ok |
| `\tokEpochs` | 500 | tokenizer epochs | setup | tokenizer checkpoint (epoch field) | ok |
| `\tokClipLen` | 8 | tokenizer clip length, frames | setup | tokenizer_cfg | ok |
| `\tokMaskMax` | 0.9 | upper end of the tokenizer mask ratio, drawn uniformly from 0 once per clip | setup | tokenizer_cfg; tokenizer/masking.py | ok |
| `\tokStepsPerEpoch` | 500 | tokenizer steps per epoch | setup | CONSTANT: SETUP_FACTS tokenizer training | ok |
| `\tokBatchEarly` | 10 | tokenizer batch size, epochs 1 to 350 | setup | CONSTANT: SETUP_FACTS tokenizer table (wandb runs of the two segments) | ok |
| `\tokBatchLate` | 32 | tokenizer batch size, epochs 351 to 500 | setup | CONSTANT: SETUP_FACTS tokenizer table (wandb run of the last segment) | ok |
| `\lpipsWeight` | 0.2 | weight of the LPIPS term in the tokenizer loss | setup | CONSTANT: tokenizer/losses.py (total = mse + 0.2 * lpips) | ok |
| `\clipTok` | 1.0 | tokenizer gradient clip | setup | tokenizer/train_tokenizer.py default | ok |
| `\lrPOne` | 3\times10^{-4} | Phase 1 peak learning rate (cosine to min-lr) | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\stepsPerEpochPOne` | 500 | Phase 1 steps per epoch | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\clipPOne` | 5 | Phase 1 gradient clip | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\wdHeavyPOne` | 0.1 | Phase 1 weight decay on the attention and feed forward matrices (0.01 on other matrices, 0 on norms and biases) | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\seqShort` | 8 | short training window, frames | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\seqLong` | 32 | long training window, frames | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\longBatchRatio` | 0.15 | share of long window batches | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\contextLength` | 16 | model context length, frames | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\registerTokens` | 4 | register tokens | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\curriculumWarmup` | 2000 | flow only steps before the bootstrap loss ramps in | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\curriculumRamp` | 4000 | global step at which the bootstrap ramp is complete; the ramp starts right after the flow only steps | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\torchPods` | 2.8.0 | PyTorch version on the rented GPUs (corrected pipeline) | C14 | requirements.txt recorded by the corrected Phase 1 run; the same value was read on 2026-10-12 from two Phase 2 and two Phase 3 runs | ok |
| `\torchLaptop` | 2.9.1 | PyTorch version on the laptop (checks and re-evaluations) | C14 | CONSTANT: sheet BU (laptop torch 2.9.1) | ok |
| `\warmupPOne` | 1000 | Phase 1 warmup steps | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\wdPOne` | 0.01 | Phase 1 weight decay on matrices other than attention and feed forward | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\lrPTwo` | 1\times10^{-4} | Phase 2 base learning rate (launch script left the default) | setup | dynamics/train_agent.py default (unchanged since c680182) | ok |
| `\warmupPTwo` | 500 | Phase 2 warmup steps (launch script left the default) | setup | dynamics/train_agent.py default | ok |
| `\minLrPTwo` | 1\times10^{-6} | Phase 2 floor of the cosine decay, before the multipliers (launch script left the default) | setup | dynamics/train_agent.py default | ok |
| `\lrMultDyn` | 0.3 | Phase 2: world model learning rate multiplier | setup | dynamics/train_agent.py default | ok |
| `\lrMultHeads` | 3 | Phase 2: heads learning rate multiplier | setup | dynamics/train_agent.py default | ok |
| `\lrPThree` | 3\times10^{-5} | Phase 3 learning rate | setup | imagination_cfg | ok |
| `\warmupPThree` | 100 | Phase 3 warmup steps | setup | imagination_cfg | ok |
| `\wdPThree` | 0.01 | Phase 3 weight decay | setup | imagination_cfg | ok |
| `\minLrPThree` | 1\times10^{-6} | Phase 3 final learning rate of the cosine decay | setup | imagination_cfg | ok |
| `\clipPThree` | 10 | Phase 3 gradient clip | setup | imagination_cfg | ok |
| `\policyBins` | 41 | bins per action dimension of the categorical policy head | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\kMax` | 64 | finest shortcut grid: smallest step size is 1/kMax | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\kSample` | 4 | denoising steps per generated frame in imagination (step size 1/kSample) | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\imagReadSignal` | 0.75 | signal level of the new frame at the last denoising step, where imagination reads the agent state ((K-1)/K) | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt); imagination/rollout.py (tau_k = k / K) | ok |
| `\dreamerEvalEpisodes` | 1000 | evaluation episodes behind each Dreamer 4 success rate (paper text) | setup | CONSTANT: arXiv 2509.24527, Table 7 caption | ok |
| `\pythonPods` | 3.12.3 | Python version on the rented GPUs (corrected Phase 1 run) | C14 | wandb-metadata.json of the corrected Phase 1 run | ok |
| `\pythonLaptop` | 3.10.19 | Python version of the laptop environment (parent evaluations and checks) | C14 | the conda environment that runs this script | ok |
| `\dmControlLaptop` | 1.0.34 | dm_control version of the laptop environment, installed 2025-11-22 and unchanged since (not recorded on the rented evaluation machines) | C14 | package metadata of the conda environment | ok |
| `\ctxSignal` | 0.9 | signal level of context frames (1 minus tau_ctx) | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\ctxFrames` | 4 | real context frames before each imagined rollout | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\horizon` | 15 | imagination horizon, steps | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\pmpoAlpha` | 0.5 | PMPO weight between positive and negative advantages | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\pmpoBeta` | 0.3 | PMPO weight of the reverse KL to the frozen BC prior | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\discount` | 0.997 | discount factor | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\lambdaRet` | 0.95 | lambda of the TD(lambda) returns | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\epochsPThree` | 15 | imagination training epochs | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\stepsPThree` | 3000 | imagination training steps in total | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\batchPThree` | 48 | imagination batch size | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\epochsPTwo` | 40 | Phase 2 finetuning epochs | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\stepsPTwo` | 20000 | Phase 2 finetuning steps in total | setup | checkpoint configs (phase2 dynamics_epoch_040.pt, phase3 final.pt) | ok |
| `\batchPTwo` | 32 | Phase 2 batch size | setup | phase2/seed*/provenance/run_phase2.sh | ok |
| `\batchPOne` | 48 | Phase 1 batch size | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\epochsPOne` | 320 | Phase 1 epochs (checkpoint used: epoch 320 of a 600-epoch cosine schedule) | setup | CONSTANT: sheet BN / AV (both Phase 1 runs stopped at epoch 320) | ok |
| `\cosineEpochsPOne` | 600 | length of the Phase 1 cosine learning rate schedule, epochs | setup | recorded arguments of the corrected Phase 1 run (wandb-metadata.json) | ok |
| `\imgSize` | 128 | image side length, pixels | setup | ball_in_cup_catch_aligned.npz frames shape | ok |
| `\episodeSteps` | 500 | control steps per episode | setup | ball_in_cup_catch_aligned.npz actions shape | ok |
| `\nTrainEps` | 216 | training episodes (the other 24 are held out) | setup | OfflineDataset split: val_fraction 0.1, split_seed 0 | ok |
| `\nExpertDemos` | 20 | expert episodes in the dataset | setup | CONSTANT: dataset README / sheet BJ | ok |
| `\nMixedSmallDemos` | 20 | episodes in the dataset's mixed-small split | setup | CONSTANT: dataset README / sheet BJ | ok |
| `\nMixedLargeDemos` | 200 | episodes in the dataset's mixed-large split | setup | CONSTANT: dataset README / sheet BJ | ok |
| `\nNoisyDemos` | 220 | episodes from noise-injected rollouts | setup | CONSTANT: dataset README / sheet BJ | ok |
| `\tokAllEpsEpochs` | 350 | tokenizer epochs trained on all 240 episodes, before the train/held-out split (epochs 351-500 used the 216 training episodes only) | setup | CONSTANT: SETUP_FACTS.md tokenizer training table (epochs 1-190 and 191-350 on all 240, 351-500 on 216) | ok |
| `\epochEarly` | 10 | the earlier checkpoint compared with the final epoch in the training-length check | C3 | CONSTANT: checkpoints saved at epochs 5, 10, 15; sheet BR addendum | ok |
| `\stopThreshold` | 2 | stopping rule: extend training if epoch 15 beats epoch 10 by more than this many points | C3 | CONSTANT: sheet BR addendum (stop rule) | ok |
| `\bootArtifactMedian` | 838 | OLD Phase 1: median raw bootstrap loss, steps 4,500-17,000 (wandb) | C7 | CONSTANT: sheet BK | ok |
| `\bootArtifactEnd` | 288600 | OLD Phase 1: raw bootstrap loss at the end of training (wandb) | C7 | CONSTANT: sheet BO | ok |
| `\dreamerMouseClasses` | 121 | Dreamer 4 (Minecraft): classes of the joint categorical for mouse movement (11 x 11) | setup | CONSTANT: Dreamer 4 paper text, 'mouse actions as a categorical with 121 classes' | ok |
| `\bootSignal` | 0.0004 | genuine raw bootstrap loss once the artifact is removed (wandb) | C7 | CONSTANT: sheet BK | ok |
