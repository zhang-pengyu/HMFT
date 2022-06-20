import os
import os.path
import numpy as np
import torch
import csv
import pandas
from collections import OrderedDict
import sys
from .base_dataset import BaseDataset
from ltr.data.image_loader import opencv_loader, jpeg4py_loader
from ltr.admin.environment import env_settings
import pdb

class UAV_RGBT(BaseDataset):

    def __init__(self, root=None, image_loader=opencv_loader, split=None, modality = None):
        if modality is None:
            raise ValueError('Unknown modality mode.')
        else:
            self.modality = modality

        root = env_settings().UAV_RGBT_dir if root is None else root
        self.root_i = env_settings().UAV_RGBT_dir
        super().__init__(root, image_loader)

        # all folders inside the root
        self.sequence_list = self._get_sequence_list()

        # seq_id is the index of the folder inside the got10k root path
        if split is not None:
            ltr_path = os.path.join(self.root_i, '..')
            if split == 'train':
                file_path = os.path.join(ltr_path,  'ST_train_split.txt')
            elif split == 'val':
                file_path = os.path.join(ltr_path, 'ST_val_split.txt')
            else:
                raise ValueError('Unknown split name.')
            sequence_list = pandas.read_csv(file_path, header=None, squeeze=True).values.tolist()
        # self.seq_ids = seq_ids
        self.init_idx = np.load(os.path.abspath(os.path.join(env_settings().UAV_RGBT_dir,'../../init_frame.npy')),allow_pickle=True).item()
        self.sequence_list = sequence_list

    def get_name(self):
        return 'UAV_RGBT'

    def has_class_info(self):
        return True

    def _load_meta_info(self):
        sequence_meta_info = {s: self._read_meta(os.path.join(self.root, s)) for s in self.sequence_list}
        return sequence_meta_info

    def _build_seq_per_class(self):
        seq_per_class = {}

        for i, s in enumerate(self.sequence_list):
            object_class = self.sequence_meta_info[s]['object_class']
            if object_class in seq_per_class:
                seq_per_class[object_class].append(i)
            else:
                seq_per_class[object_class] = [i]

        return seq_per_class

    def _get_parent_classList(self):
        project_path = os.path.abspath(os.path.dirname(__file__))
        file_path = os.path.join(project_path, '../data_specs/parent_class_imagenet_extended.txt')

        # load the parent class file -> refer to the imagenet website for the list of parent classes
        f = open(file_path)
        major_classes = list(csv.reader(f))
        f.close()

        parent_classes = [cls[0] for cls in major_classes]
        return parent_classes

    def get_sequences_in_class(self, class_name):
        return self.seq_per_class[class_name]

    def _get_sequence_list(self):

        return os.listdir(self.root)

    def _read_bb_anno(self, seq_path):
        if self.modality in ['RGB', 'RGBT']: 
            bb_anno_file = os.path.join(seq_path, "rgb.txt")
            gt = np.loadtxt(bb_anno_file).astype(np.float32)
        elif self.modality in ['T']:
            bb_anno_file = os.path.join(seq_path, "ir.txt")
            gt = np.loadtxt(bb_anno_file).astype(np.float32)
        else:
            raise ValueError('Unknown modality mode.')

        return torch.tensor(gt)

    def _read_target_visible(self, seq_path):
        # Read full occlusion and out_of_view
        occlusion_file = os.path.join(seq_path, "absence.label")
        cover_file = os.path.join(seq_path, "cover.label")

        with open(occlusion_file, 'r', newline='') as f:
            occlusion = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])
        with open(cover_file, 'r', newline='') as f:
            cover = torch.ByteTensor([int(v[0]) for v in csv.reader(f)])

        target_visible = ~occlusion & (cover>0)

        return target_visible

    def _get_sequence_path_i(self, seq_id):
        return os.path.join(self.root_i, self.sequence_list[seq_id])

    def _get_sequence_path(self, seq_id):
        return os.path.join(self.root, self.sequence_list[seq_id])

    def get_sequence_info(self, seq_id):
        seq_path = self._get_sequence_path(seq_id)
        bbox = self._read_bb_anno(seq_path)

        valid = (bbox[:, 2] > 0) & (bbox[:, 3] > 0)

        visible = valid

        return {'bbox': bbox, 'valid': valid, 'visible': visible}

    def _get_frame_path(self, seq_path, modality, frame_id):
        seq_name = seq_path.split('/')[-1] 
        if seq_name in self.init_idx:
            init_idx = self.init_idx[seq_name]
        else:
            init_idx = 0
        nz = 6
        return os.path.join(seq_path, modality, str(frame_id*10+init_idx).zfill(nz)+'.jpg')    # frames start from 1

    def _get_frame(self, seq_path, modality, frame_id):
        return self.image_loader(self._get_frame_path(seq_path, modality, frame_id))

    def get_class_name(self, seq_id):
        obj_meta = self.sequence_meta_info[self.sequence_list[seq_id]]

        return obj_meta['object_class']

    def get_frames(self, seq_id, frame_ids, anno=None):
        
        if self.modality in ['RGB']:
            seq_path = self._get_sequence_path(seq_id)
            frame_list = [self._get_frame(seq_path, 'rgb', f_id) for f_id in frame_ids]

        elif self.modality in ['T']:
            seq_path_i = self._get_sequence_path_i(seq_id)
            frame_list = [self._get_frame(seq_path_i,'ir_gray', f_id) for f_id in frame_ids]
        elif self.modality in ['RGBT']:
            seq_path = self._get_sequence_path(seq_id)
            seq_path_i = self._get_sequence_path_i(seq_id)
            frame_list_v = [self._get_frame(seq_path, 'rgb', f_id) for f_id in frame_ids]
            frame_list_i = [self._get_frame(seq_path_i,'ir_gray', f_id) for f_id in frame_ids]
            frame_list = frame_list_v + frame_list_i
        
        if anno is None:
            anno = self.get_sequence_info(seq_id)
        
        anno_frames = {}
        for key, value in anno.items():
            anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids]
            #anno_frames[key] = [value[f_id, ...].clone() for f_id in frame_ids] + [value[f_id, ...].clone() for f_id in frame_ids]
        
        return frame_list, anno_frames
