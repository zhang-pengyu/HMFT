import torch
from torch import nn
import torch.nn.functional as F
from collections import OrderedDict
import numpy as np
from torch.nn.modules.pooling import AdaptiveAvgPool2d

class Late_fusion(nn.Module):
    def __init__(self, feat_dim = 1024):
        super().__init__()

        self.FF_layers = nn.Sequential(OrderedDict([                 
                ('conv1_dis',   nn.Sequential(
                                        nn.Conv2d(1024,256,kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv2_dis',   nn.Sequential(
                                        nn.Conv2d(1024,256,kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        )),
                
                ('conv1_comp',   nn.Sequential(
                                        nn.Conv2d(2048,256,kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        )),
                ('conv2_comp',   nn.Sequential(
                                        nn.Conv2d(2048,256,kernel_size=1,stride=1),
                                        nn.ReLU(),
                                        )),

                ('encoder_decoder',   nn.Sequential(
                                        nn.Conv2d(2,16,kernel_size=3,stride=1,dilation=2),
                                        nn.ReLU(),
                                        nn.ConvTranspose2d(16,2,kernel_size=3,stride=1, dilation=2),
                                        nn.ReLU()
                                        ))
                                        ]))

    def forward(self,response_dis, response_comp, feat_dis, feat_comp, cls_loss_RGB = None, cls_loss_T = None):
        if len(feat_dis.shape) == 4:
            feat_dis = feat_dis.unsqueeze(0)
            feat_comp = feat_comp.unsqueeze(0)
            # response_dis = response_dis.unsqueeze(0)
            # response_comp = response_dis.unsqueeze(0)
            
        feat_S, feat_L, feat_C, feat_H, feat_W = feat_dis.shape
        resp_S, resp_L, resp_H, resp_W = response_dis.shape

        dis1 = self.FF_layers.conv1_dis(feat_dis.view(feat_S*feat_L,feat_C, feat_H, feat_W)).view(feat_S*feat_L,256,-1)
        dis2 = self.FF_layers.conv2_dis(feat_dis.view(feat_S*feat_L,feat_C, feat_H, feat_W)).view(feat_S*feat_L,256,-1).permute(0,2,1)
        simi_matrix = torch.bmm(dis2,dis1)
        simi_matrix_dis = simi_matrix.clone()

        dis_reshape = feat_dis.view(feat_S*feat_L, feat_C, feat_H*feat_W)
        dis = torch.sum(torch.bmm(dis_reshape,simi_matrix),dim=1).view(feat_S, feat_L, 1, feat_H, feat_W)

        comp1 = self.FF_layers.conv1_comp(feat_comp.view(feat_S*feat_L, feat_C*2, feat_H, feat_W)).view(feat_S*feat_L,256,-1)
        comp2 = self.FF_layers.conv2_comp(feat_comp.view(feat_S*feat_L, feat_C*2, feat_H, feat_W)).view(feat_S*feat_L,256,-1).permute(0,2,1)
        simi_matrix = torch.bmm(comp2,comp1)
        simi_matrix_comp = simi_matrix.clone()

        comp_reshape = feat_comp.view(feat_S*feat_L, feat_C*2, feat_H*feat_W)
        comp = torch.sum(torch.bmm(comp_reshape,simi_matrix),dim=1).view(feat_S, feat_L, 1, feat_H, feat_W)

        '''type 1'''
        weight = torch.cat((dis,comp),dim=2)
        weight_tmp = weight.clone()
        weight = torch.nn.functional.layer_norm(weight,(weight.shape[0],weight.shape[1],weight.shape[2],weight.shape[3],weight.shape[4]))
        '''type 2'''
        # dis = (dis - dis.min())/(dis.max()- dis.min())
        # comp = (comp - comp.min())/(comp.max()- comp.min())
        # weight = torch.cat((dis,comp),dim=2)
        
        weight_tmp = weight.clone()
        weight = self.FF_layers.encoder_decoder(weight.view(feat_S*feat_L, 2, feat_H, feat_W))
        weight = weight.view(feat_S, feat_L, 2, feat_H, feat_W) 
        weight = nn.functional.softmax(nn.functional.interpolate(weight,size=(2, resp_H, resp_W), mode = 'nearest'),dim=2)

        # import matplotlib.pyplot as plt 
        # fig, axs = plt.subplots(8)
        # axs[0].imshow(feat_dis[0,0,0].cpu().detach())
        # axs[1].imshow(dis[0,0,0].cpu().detach())
        # axs[2].imshow(response_dis[0,0].cpu().detach())
        # axs[3].imshow(weight_tmp[0,0,0].cpu().detach())
        # axs[4].imshow(feat_comp[0,0,0].cpu().detach())
        # axs[5].imshow(comp[0,0,0].cpu().detach())
        # axs[6].imshow(response_comp[0,0].cpu().detach())
        # axs[7].imshow(weight_tmp[0,0,1].cpu().detach())
        # plt.show()

        response = response_dis * weight[:,:,0,:,:] + response_comp * weight[:,:,1,:,:]

        if cls_loss_RGB is not None:
            losses = {'train': [], 'test': []} 
            losses['train'] = list(np.array(cls_loss_RGB['train'] + cls_loss_T['train'])/2)
            losses['test'] = list(np.array(cls_loss_RGB['test'] + cls_loss_T['test'])/2)
        else:
            losses = None 
        return response, losses
        
