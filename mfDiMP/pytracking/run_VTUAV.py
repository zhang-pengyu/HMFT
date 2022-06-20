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
    
    os.environ["CUDA_VISIBLE_DEVICES"] ="0"

    dataset_path = '' # path for dataset 
    seq_home = dataset_path
    seq_list = [f for f in os.listdir(seq_home) if os.path.isdir(os.path.join(seq_home,f))]
    seq_list.sort()

        
    
    tracker = Tracker('etcom_comb2', 'debug_lichao_comb23_2')
    
    tracker.parameters.features.features[0].dis_net_path = 'OptimTracker_ep0045.pth.tar'
    for name in seq_list:

        save_path = './results_VTUAV/' + name + '.txt'
        save_folder = './results_VTUAV/' 
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.exists(save_path):
            continue
        if len(seq_list) == len(os.listdir(save_folder)):
            break

        print('——————————Process sequence: '+ name +'——————————————')

        seq = dict()

        rgb_path = seq_home + '/' + name + '/rgb'
        t_path = seq_home + '/' + name + '/ir_gray'
        rgb_dir = glob(rgb_path + '/*.jpg')
        rgb_dir.sort()

        t_dir = glob(t_path+ '/*.jpg')
        t_dir.sort()

        rgb_gt = np.loadtxt(seq_home + '/' + name + '/rgb.txt')
        t_gt = np.loadtxt(seq_home + '/' + name + '/ir.txt')

        seq['rgb_frame'] = rgb_dir
        seq['t_frame'] = t_dir
        seq['rgb_gt'] = rgb_gt
        seq['t_gt'] = t_gt
        
        bb, time = tracker.run(seq,visualization=False)
        np.savetxt(save_path,bb)
        print('fps:  ' + str(1/np.mean(time)))
        
