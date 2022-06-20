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
    tracker.parameters.features.features[0].dis_net_path = 'OptimTracker_ep'+str(45).zfill(4)+'.pth.tar'
    
    '''setting'''
    tracker.parameters.iounet_k = 8
    tracker.parameters.maximal_aspect_ratio = 8
    tracker.parameters.search_area_scale = 4

    tracker.parameters.sample_memory_size = 150
    tracker.parameters.image_sample_size = 20 * 16
    tracker.parameters.net_opt_iter = 20

    tracker.parameters.train_skipping = 30
    tracker.parameters.net_opt_update_iter = 5
    tracker.parameters.target_not_found_threshold = 0.01

    tracker.parameters.net_opt_hn_iter = 1
    tracker.parameters.target_neighborhood_scale = 2.2
    tracker.parameters.distractor_threshold = 100

    for name in seq_list:

        save_path = './results_RGBT234/' + name + '.txt'
        save_folder = './results_RGBT234/'
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        if os.path.exists(save_path):
            continue
        if len(seq_list) == len(os.listdir(save_folder)):
            break

        print('——————————Process sequence: '+ name +'——————————————')

        seq = dict()

        rgb_path = seq_home + '/' + name + '/visible'
        t_path = seq_home + '/' + name + '/infrared'
        rgb_dir = glob(rgb_path + '/*.jpg')
        rgb_dir.sort()

        t_dir = glob(t_path+ '/*.jpg')
        t_dir.sort()

        rgb_gt = np.loadtxt(seq_home + '/' + name + '/visible.txt',delimiter=',')
        
        t_gt = rgb_gt
        
        seq['rgb_frame'] = rgb_dir
        seq['t_frame'] = t_dir
        seq['rgb_gt'] = rgb_gt
        seq['t_gt'] = t_gt
        # try:
        bb, time = tracker.run(seq,visualization=False)
        np.savetxt(save_path,bb)
        print('fps:  ' + str(1/np.mean(time)))
