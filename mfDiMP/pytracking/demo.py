#!/usr/bin/env python
import os
import sys
from glob import glob 
import numpy as np
import torch

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)
from pytracking.evaluation import Tracker

if __name__ == "__main__":
    
    os.environ["CUDA_VISIBLE_DEVICES"] ="1"
    seq_name = 'bus1_sample' # other test sequences can be downloaded via gdown in "install.sh"
    seq_path = os.path.abspath(env_path +'/demo/'+seq_name)

    tracker = Tracker('etcom_comb2', 'debug_lichao_comb23_2')
    tracker.parameters.features.features[0].dis_net_path = 'OptimTracker_ep'+str(45).zfill(4)+'.pth.tar'

    save_path = './results/demo.txt'
    save_folder = './results/'
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
            
    print('——————————Process sequence: demo——————————————')

    seq = dict()

    rgb_path = seq_path + '/rgb'
    t_path = seq_path +'/ir_gray'
    rgb_dir = glob(rgb_path + '/*.jpg')
    rgb_dir.sort()

    t_dir = glob(t_path+ '/*.jpg')
    t_dir.sort()

    rgb_gt = np.loadtxt(seq_path + '/rgb.txt')
    t_gt = np.loadtxt(seq_path + '/ir.txt')

    seq['rgb_frame'] = rgb_dir
    seq['t_frame'] = t_dir
    seq['rgb_gt'] = rgb_gt
    seq['t_gt'] = t_gt
    
    bb, time = tracker.run(seq,visualization=True)
    np.savetxt(save_path,bb)
    print('fps:  ' + str(1/np.mean(time)))
    
