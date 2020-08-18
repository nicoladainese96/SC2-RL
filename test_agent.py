import os
import numpy as np

names = np.array(['CollectMineralShards%s'%i for i in range(1,6)]+\
        ['FindAndDefeatZerglings%s'%i for i in range(1,6)])

for n in names:
    cmd = 'python monobeast.py --map_name %s --mode test --savedir ~/Desktop/Master\ Thesis/SC2-RL/logs/torchbeast/ --xpid %s'%(n[:-1],n)
    os.system(cmd)