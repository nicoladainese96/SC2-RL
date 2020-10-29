# SC2-RL

Last update: 29/10/2020

Repository containing the code for my Master's Thesis **"Deep Reinforcement Learning for StarCraft II Learning Environment"**, developed during internship at Aalto University under the supervision of Alexander Ilin and Rinu Boney.

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

## Future developments and known issues

Here there's a list of what I would have liked to do but I didn't have either the time or the resources to do it. 
Also I report some issues of which I'm aware. 

Since I'm not currently working on this project anymore, I might not implement these points and necessary fixes myself, but I'm open to collaborations, questions, explanations and so on, so feel free to open an issue if needed.

**Possible developments:**
1. Adapt the code from polybeast (https://github.com/facebookresearch/torchbeast) to the StarCraft II Learning Environment (SC2LE). This should enable multi-machine training and should use the GPU device also in the forward pass, similarly as what is done in the GA3C architecture (https://arxiv.org/abs/1611.06256)
2. Train the Fully Convolutional agent on the last 3 minigames and see how long it takes to get decent results (I would say certainly more than 3 days with my setup).
3. Train the control architecture from the paper **"Relational Deep Reinforcement Learning"** from DeepMind (https://arxiv.org/abs/1806.01830). This is already implemented in monobeast_v2.py (see from there all modules that are imported/different from monobeast.py), altough there is an issue with the action mask for some minigames (discussed more in depth below). I was able to train this architecture for CollectMineralShards for 6M steps (see Chapter 7.4 of the thesis), getting results similar to the one with the shallow Fully Convolutional architecture, but I expect that much better results can be obtained with more compute and a little bit of fine tuning on the entropy cost and learning rate.
4. Implement and train the relational architecture from **"Relational Deep Reinforcement Learning"** and replicate the results. This actually I think would be very expensive, since they used 10 billions steps for each run, which is approximately 1000 times more the steps than the ones I was able to get with 3 compute days and the code from monobeast.

**Known issues:**
1. On monobeast_v2.py the spatial inputs have to be modified by adding a binary mask layer containing all ones if the last action was applied to that spatial space (e.g. if the last action was MoveScreen, we add a layer of ones to the screen state and a layer of zeros to the minimap state; if it was MoveMinimap we do the opposite; finally if it was a non-spatial action like SelectArmy, we add a binary mask of zeros both to the screen and the minimap states). The problem here as I understand it is that I'm restricting the action space to approximately 20 actions and my pre-processing pipeline is able to recognize if an action is applied to the screen, the minimap or neither of them only for these actions; however sometimes when the agent selects an action, the action gets translated to another action before being passed to the environment, so the next time-step the last action variable can contain actions out of the action space selected. This could be solved by setting to no-op every action out of the selected action space, but I still have to do that. One minigame in which the problem can be reproduced is FindAndDefeatZerglings.




