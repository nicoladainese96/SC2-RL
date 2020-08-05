# SC2-RL

Last update: 24/07/2020

Repository containing the code for my master thesis Deep Reinforcement Learning for StarCraft II Learning Environment, developed during internship at Aalto University under the supervision of Alexander Ilin and Rinu Boney.

**tl;dr**: Adapted "Importance Weighted Actor-Learner Architecture" (IMPALA) from https://github.com/facebookresearch/torchbeast to StarCraft II Learning Environment (https://github.com/deepmind/pysc2). Almost 10 times faster than A2C (also implemented here, but almost from scratch), works on all 7 minigames. Code in PyTorch.

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

## Preliminary results

MoveToBeacon: 120k steps, average asymptotic score 26 (target to reach\*: 26)
CollectMineralShards: 3.6M steps, average asymptotic score 94 (target to reach\*: 103)
FindAndDefeatZerglings: 3.6M steps, average asymptotic score 43 (target to reach\*: 45)

\*based on results from StarCraft II: A New Challenge for Reinforcement Learning (https://arxiv.org/abs/1708.04782) for the FullyConv agent.

<img src='Supplementary material/MTB.png'>
<img src='Supplementary material/CMS.png'>
<img src='Supplementary material/FADZ.png'>