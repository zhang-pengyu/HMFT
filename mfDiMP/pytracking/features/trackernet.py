from pytracking.features.featurebase import FeatureBase, MultiFeatureBase
import torch
from collections import OrderedDict
import torchvision
import importlib
from pytracking import TensorList
from pytracking.evaluation.environment import env_settings
from pytracking.utils.loading import load_network, load_network_zpy
import os


class SimpleTrackerResNet18(MultiFeatureBase):
    def __init__(self, dis_net_path, comp_net_path = None, output_layers=None, use_gpu=True, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # for l in output_layers:
        #     if l not in ['vggconv1', 'conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'fc']:
        #         raise ValueError('Unknown layer')

        self.dis_net_path = dis_net_path
        self.comp_net_path = comp_net_path
        self.output_layers = ['classification'] if output_layers is None else output_layers
        self.use_gpu = use_gpu

    def initialize(self):
        if self.comp_net_path is not None:
            self.net = load_network_zpy(self.dis_net_path,self.comp_net_path)
        else:
            self.net = load_network(self.dis_net_path)
            
        if self.use_gpu:
            self.net.cuda()
        self.net.eval()

        self.iou_predictor = self.net.bb_regressor
        self.target_classifier = self.net.classifier
        self.target_classifier_vi = self.net.classifier_vi
        
        self.layer_stride = {'vggconv1': 2, 'conv1': 2, 'layer1': 4, 'layer2': 8, 'layer3': 16, 'layer4': 32, 'classification': 16, 'fc': None}
        self.layer_dim = {'vggconv1': 96, 'conv1': 64, 'layer1': 64, 'layer2': 128, 'layer3': 256, 'layer4': 512, 'classification': 256,'fc': None}

        self.iounet_feature_layers = self.net.bb_regressor_layer
        self.classification_feature_layer = self.net.classification_layer

        if hasattr(self.net, 'center_regressor'):
            self.center_regressor = self.net.center_regressor

        # if self.output_layers is None:
        #     self.output_layers = [self.classification_feature_layer]

        if isinstance(self.pool_stride, int) and self.pool_stride == 1:
            self.pool_stride = [1]*len(self.output_layers)

        # all_layers = ['vggconv1', 'conv1', 'layer1', 'layer2', 'layer3']
        self.feature_layers = sorted(list(set(self.output_layers + self.iounet_feature_layers)))

        self.mean = torch.Tensor([0.485, 0.456, 0.406]).view(1,-1,1,1)
        self.std = torch.Tensor([0.229, 0.224, 0.225]).view(1,-1,1,1)

    def free_memory(self):
        if hasattr(self, 'net'):
            del self.net
        if hasattr(self, 'iou_predictor'):
            del self.iou_predictor
        if hasattr(self, 'target_classifier'):
            del self.target_classifier
        if hasattr(self, 'iounet_backbone_features'):
            del self.iounet_backbone_features
        if hasattr(self, 'iounet_features'):
            del self.iounet_features
        if hasattr(self, 'center_regressor'):
            del self.center_regressor

    def dim(self):
        return TensorList([self.layer_dim[l] for l in self.output_layers])

    def stride(self):
        return TensorList([s * self.layer_stride[l] for l, s in zip(self.output_layers, self.pool_stride)])

    def extract(self, im: torch.Tensor):
        im = im/255
        im -= self.mean
        im /= self.std

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            output_features = self.net.extract_features(im, self.feature_layers)

        # Store the raw resnet features which are input to iounet
        self.iounet_backbone_features = TensorList([output_features[layer].clone() for layer in self.iounet_feature_layers])

        # Store the processed features from iounet, just before pooling
        with torch.no_grad():
            self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))

        return TensorList([output_features[layer] for layer in self.output_layers])

    def extract_comb(self, im: torch.Tensor):
        im = im[:,:6,...]/255
        #im -= self.mean
        #im /= self.std
        im -= torch.cat((self.mean, self.mean), 1)
        im /= torch.cat((self.std, self.std), 1)

        if self.use_gpu:
            im = im.cuda()

        with torch.no_grad():
            output_features = self.net.extract_features(im, self.feature_layers)
            output_features_vi = self.net.extract_features_vi(im,self.feature_layers)

            self.feat_dis = self.net.feat_dis
            self.feat_comp = self.net.feat_comp

        # Store the raw resnet features which are input to iounet
        self.iounet_backbone_features = TensorList([torch.cat((output_features[layer].clone(),output_features_vi[layer].clone()),dim=1) for layer in self.iounet_feature_layers])

        # Store the processed features from iounet, just before pooling
        with torch.no_grad():
            self.iounet_features = TensorList(self.iou_predictor.get_iou_feat(self.iounet_backbone_features))

        return TensorList([output_features[layer] for layer in self.output_layers]), TensorList([output_features_vi[layer] for layer in self.output_layers])

    