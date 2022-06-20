import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict

from torch.nn.modules.pooling import AdaptiveAvgPool2d

class Feature_fusion(nn.Module):
    def __init__(self, feat_dim, hidden_dim = 256):
        super().__init__()
        self.GAP = nn.AdaptiveAvgPool2d(1)
        self.FF_layers = nn.Sequential(OrderedDict([                 
                ('fc1',   nn.Sequential(
                                        nn.Linear(feat_dim, hidden_dim),
                                        nn.ReLU())),
                ('fc2_RGB',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, feat_dim),
                                        nn.ReLU())),
                ('fc2_T',   nn.Sequential(nn.Dropout(0.5),
                                        nn.Linear(hidden_dim, feat_dim),
                                        nn.ReLU()))

                                        ]))

    def forward(self,feat_RGB,feat_T):
        feat_sum = self.GAP(feat_RGB+feat_T)
        feat_sum = feat_sum.view(-1, feat_sum.shape[1])
        feat_sum = self.FF_layers.fc1(feat_sum)
        w_RGB = self.FF_layers.fc2_RGB(feat_sum)
        w_T = self.FF_layers.fc2_T(feat_sum)
        w = nn.functional.softmax(torch.cat([w_RGB, w_T], 0),dim=0)

        feat = feat_RGB * w[0,:].view(1,w.size()[1],1,1) + feat_T * w[1,:].view(1,w.size()[1],1,1)
        return feat
        
