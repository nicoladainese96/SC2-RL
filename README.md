# SC2-RL

Last update: 29/10/2020

Repository containing the code for my Master's Thesis "Deep Reinforcement Learning for StarCraft II Learning Environment", developed during internship at Aalto University under the supervision of Alexander Ilin and Rinu Boney.

**tl;dr**: Adapted "Importance Weighted Actor-Learner Architecture" (IMPALA) from https://github.com/facebookresearch/torchbeast to StarCraft II Learning Environment (https://github.com/deepmind/pysc2). Approximately 16 times faster than A2C (also implemented here, but almost from scratch), works on all 7 minigames. Code in PyTorch. Pdf of the thesis available in the repository.

## Fast run

To run the training with default options:
> python monobeast.py 

To run in test mode
1. copy xpid from final output of the respective training; 
2. specify again the map name (if different from default) or it will run on MoveToBeacon as default
> python monobeast.py --mode test --xpid \< xpid \>

Notebook to inspect the results at Notebooks/Part21_IMAPALA_Results.

Note: at the moment the steps are counted as number of optimizer steps x number of timesteps per trajectory, so it's NOT multiplied by the batch size.

Main parameters to play with:
- `--map_name` (one in MoveToBeacon, CollectMineralShards, FindAndDefeatZerglings, DefeatRoaches, DefeatZerglingsAndBanelings, CollectMineralsAndGas, BuildMarines)
- `--num_actors` (I usually use same number as the number of CPU cores)
- `--batch_size` (keep it 2 times num_actors to be safe)
- `--total_steps` (I will publish the minimum number of steps to train on each minigame together with the values of the parameters that I changed as soon as I run the experiments)

## Requirements
Main requirements:
- pysc2 (pip install pysc2 or see https://github.com/deepmind/pysc2)
- StarCraftII (see https://github.com/Blizzard/s2client-proto#downloads to install it on Linux; currently using version 4.10)
- pytorch (tested in version 1.5.0 / 1.5.1)

## A2C results

<img src='Supplementary material/A2C-results.png'>

<img src='Supplementary material/MTB-A2C.png'>
<img src='Supplementary material/CMS-A2C.png'>
<img src='Supplementary material/FADZ-A2C.png'>
<img src='Supplementary material/DZAB-A2C.png'>

## IMPALA results

<img src='Supplementary material/IMPALA-results.png'>

<img src='Supplementary material/MTB-IMPALA.png'>
<img src='Supplementary material/CMS-IMPALA.png'>
<img src='Supplementary material/FADZ-IMPALA.png'>
<img src='Supplementary material/DZAB-IMPALA.png'>

## Conclusions

Before comparing our results to the ones from DeepMind, it is worth noting that their agents were trained for 600 millions steps with the equivalent of a batch size of 64, which is around 2 orders of magnitude more than the amount of training that we used. Moreover they reported only the average results achieved by the best out of 100 runs: this implies that their results should be compared only with our best run average score, but even then the different number of runs rig the comparison in favour of DeepMind score.

Looking at the comparison between our results with IMPALA and the DeepMind FullyConv agent we notice that:

- The performance of MoveToBeacon is slightly lower, a gap which could be filled with a tuning in the hyper-parameters (mainly the learning rate and entropy cost).
- The performance of CollectMineralShards is significantly lower, on average because one run failed to learn the same strategy of the others, but also because probably the other 4 runs were using the right strategy, but without being able to optimize it properly. It might have been the case that a longer training time could have yielded better results.
- On FindAndDefeatZerglings all runs score fairly similar and the average performance is compatible with the DeepMind best mean.
- Finally on DefeatZerglingsAndBanelings there is a lot of variance in the IMPALA's results, with three runs that scored less than DeepMind best mean, one that was compatible with it and the best one that significantly outperformed it after 6 millions of steps, probably learning a qualitatively superior strategy.


If then we consider the advantage given by approximately two orders of magnitude more of training and by taking the maximum over 100 runs instead than over just 5, it is fair to expect that the IMPALA agent under the same conditions would match or surpass DeepMind best agent performance on all 4 minigames.



