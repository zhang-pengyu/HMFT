from . import BaseActor
import torch
import torch.nn.functional as F


class OptimTrackerActor(BaseActor):
    def __init__(self, net, objective, loss_weight=None):
        super().__init__(net, objective)
        if loss_weight is None:
            loss_weight = {'iou': 1.0, 'test_clf': 1.0, 'train_clf': 1.0, 'init_clf': 1.0}
        self.loss_weight = loss_weight

    def __call__(self, data):

        # Run network
        target_scores, iou_pred, clf_losses, train_feat_vi, test_feat_vi = self.net(data['train_images'],
                                                       data['test_images'],
                                                       data['train_anno'],
                                                       data['test_proposals'],
                                                       data['train_label'],
                                                       data['is_distractor_train_frame'],
                                                       test_label=data['test_label'],
                                                       test_anno=data['test_anno'])
        '''for training CIF: the definition of L_div '''
        # train_feat1_v = train_feat_vi[0]['layer1']
        # train_feat1_v = F.log_softmax(train_feat1_v.view(train_feat1_v.shape[0],train_feat1_v.shape[1],-1),dim=2)
        # train_feat2_v = train_feat_vi[0]['layer2']
        # train_feat2_v = F.log_softmax(train_feat2_v.view(train_feat2_v.shape[0],train_feat2_v.shape[1],-1),dim=2)
        # train_feat3_v = train_feat_vi[0]['layer3']
        # train_feat3_v = F.log_softmax(train_feat3_v.view(train_feat3_v.shape[0],train_feat3_v.shape[1],-1),dim=2)

        # train_feat1_i = train_feat_vi[1]['layer1']
        # train_feat1_i = F.log_softmax(train_feat1_i.view(train_feat1_i.shape[0],train_feat1_i.shape[1],-1),dim=2)
        # train_feat2_i = train_feat_vi[1]['layer2']
        # train_feat2_i = F.softmax(train_feat2_i.view(train_feat2_i.shape[0],train_feat2_i.shape[1],-1),dim=2)
        # train_feat3_i = train_feat_vi[1]['layer3']
        # train_feat3_i = F.softmax(train_feat3_i.view(train_feat3_i.shape[0],train_feat3_i.shape[1],-1),dim=2)
        
        # test_feat1_v = train_feat_vi[0]['layer1']
        # test_feat1_v = F.log_softmax(test_feat1_v.view(test_feat1_v.shape[0],test_feat1_v.shape[1],-1),dim=2)
        # test_feat2_v = train_feat_vi[0]['layer2']
        # test_feat2_v = F.log_softmax(test_feat2_v.view(test_feat2_v.shape[0],test_feat2_v.shape[1],-1),dim=2)
        # test_feat3_v = train_feat_vi[0]['layer3']
        # test_feat3_v = F.log_softmax(test_feat3_v.view(test_feat3_v.shape[0],test_feat3_v.shape[1],-1),dim=2)

        # test_feat1_i = train_feat_vi[1]['layer1']
        # test_feat1_i = F.softmax(test_feat1_i.view(test_feat1_i.shape[0],test_feat1_i.shape[1],-1),dim=2)
        # test_feat2_i = train_feat_vi[1]['layer2']
        # test_feat2_i = F.softmax(test_feat2_i.view(test_feat2_i.shape[0],test_feat2_i.shape[1],-1),dim=2)
        # test_feat3_i = train_feat_vi[1]['layer3']
        # test_feat3_i = F.softmax(test_feat3_i.view(test_feat3_i.shape[0],test_feat3_i.shape[1],-1),dim=2)

        
        # RMI_loss = self.loss_weight['RMI_loss'] * (
        # self.objective['RMI_loss'](train_feat1_v,train_feat1_i)+ \
        # self.objective['RMI_loss'](train_feat2_v,train_feat2_i)+ \
        # self.objective['RMI_loss'](train_feat3_v,train_feat3_i)+ \
        # self.objective['RMI_loss'](test_feat1_v,test_feat1_i)+ \
        # self.objective['RMI_loss'](test_feat2_v,test_feat2_i)+ \
        # self.objective['RMI_loss'](test_feat3_v,test_feat3_i))
        
        RMI_loss = torch.tensor(0).cuda()

        clf_loss_test = self.objective['test_clf'](target_scores, data['test_label'], data['test_anno'])

        is_distractor_test = data['is_distractor_test_frame'].view(-1)

        iou_pred_valid = iou_pred.view(-1, iou_pred.shape[2])[is_distractor_test == 0, :]
        iou_gt_valid = data['proposal_iou'].view(-1, data['proposal_iou'].shape[2])[is_distractor_test == 0, :]

        # Compute loss
        loss_iou = self.loss_weight['iou'] * self.objective['iou'](iou_pred_valid, iou_gt_valid)
        loss_target_classifier = self.loss_weight['test_clf'] * clf_loss_test
        loss_init_clf = self.loss_weight['init_clf'] * clf_losses['train'][0]
        loss_train_clf = self.loss_weight['train_clf'] * clf_losses['train'][-1]

        loss_iter_clf = 0
        if 'iter_clf' in self.loss_weight.keys():
            loss_iter_clf = (self.loss_weight['iter_clf'] / (len(clf_losses['train']) - 2)) * sum(clf_losses['train'][1:-1])

        loss_test_init_clf = 0
        if 'test_init_clf' in self.loss_weight.keys():
            loss_test_init_clf = self.loss_weight['test_init_clf'] * clf_losses['test'][0]

        loss_test_iter_clf = 0
        if 'test_iter_clf' in self.loss_weight.keys():
            loss_test_iter_clf = (self.loss_weight['test_iter_clf'] / (len(clf_losses['test']) - 2)) * sum(clf_losses['test'][1:-1])

        loss = loss_iou + loss_target_classifier + loss_init_clf + loss_train_clf + loss_iter_clf + loss_test_init_clf + loss_test_iter_clf

        stats = {'Loss/total': loss.item(),
                 'Loss/iou': loss_iou.item(),
                 'Loss/target_clf': loss_target_classifier.item(),
                 'Loss/init_clf': loss_init_clf.item(),
                 'Loss/train_clf': loss_train_clf.item(),
                 'Loss/RMI_loss': RMI_loss.item()
                 }

        if 'iter_clf' in self.loss_weight.keys():
            stats['Loss/iter_clf'] = loss_iter_clf.item()
        if 'test_init_clf' in self.loss_weight.keys():
            stats['Loss/test_init_clf'] = loss_test_init_clf.item()
        if 'test_iter_clf' in self.loss_weight.keys():
            stats['Loss/test_iter_clf'] = loss_test_iter_clf.item()

        stats['ClfTrain/init_loss'] = clf_losses['train'][0].item()
        stats['ClfTrain/train_loss'] = clf_losses['train'][-1].item()
        if len(clf_losses['train']) > 2:
            stats['ClfTrain/iter_loss'] = sum(clf_losses['train'][1:-1]).item() / (len(clf_losses['train']) - 2)

        stats['ClfTrain/test_loss'] = clf_loss_test.item()

        if len(clf_losses['test']) > 0:
            stats['ClfTrain/test_init_loss'] = clf_losses['test'][0].item()
            if len(clf_losses['test']) > 2:
                stats['ClfTrain/test_iter_loss'] = sum(clf_losses['test'][1:-1]).item() / (len(clf_losses['test']) - 2)

        return loss, stats

