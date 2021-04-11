from functools import partial

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from mtrain.measurer import AcuuMeasurer, CWAcuuMeasurer
from mtrain.partial_lr_scheduler import pMultiStepLR, pStepLR
from mtrain.partial_optimzer import pAdam, pSGD

from ..basic_model import TrainableModel
from .params import params

import numpy as np
from math import factorial
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# 核心类
class ZZZ_model(TrainableModel):
    def __init__(self):
        #
        # print("woereiwryiuewyr")
        self.sample_number = 5
        super().__init__(params)
        self.writer.log_asset('mmodel/ZZZ_model/networks/network.py')
        self.writer.log_asset('mmodel/ZZZ_model/model.py')
        # the loss functions needed for training
        # self.CEL = torch.nn.CrossEntropyLoss()

    def prepare_dataloaders(self):
        from .datas.rgb_feats_dataset import get_data_list, VideoDataset
        ls_s_train, ls_t_train, ls_t_eval = get_data_list(
            self.cfg.dataset, self.cfg.source, self.cfg.target)
        # constructing datasets
        _VideoDataset = partial(VideoDataset, sampled_frame=self.cfg.sampled_frame)
        source_dset = _VideoDataset(ls_s_train)
        target_dset = _VideoDataset(ls_t_train)
        eval_dset = _VideoDataset(ls_t_eval)
        assert len(source_dset) >= self.cfg.batch_size
        assert len(target_dset) >= self.cfg.batch_size
        # constructing dataloaders
        _DataLoader = partial(DataLoader, batch_size=self.cfg.batch_size,
                              shuffle=False, num_workers=4, pin_memory=False)

        train_loader = {
            'source': _DataLoader(source_dset, drop_last=False, sampler=source_dset.sampler),
            'target': _DataLoader(target_dset, drop_last=False, shuffle=True),
        }
        eval_loader = _DataLoader(eval_dset, drop_last=False)
        # set measure meter for eval dataset
        measurer = [AcuuMeasurer(), CWAcuuMeasurer()]
        return train_loader, eval_loader, measurer

    def regist_networks(self):
        from .networks.network import FeatureEncoder, Classifier, ResidualDialConvBlock, DiscrminiatorBlock, RelationModuleMultiScale, RelationModule
        from .loss import DomainBCE, HLoss, self_loss
        RESNET_DIM = 2048
        bottleneck = self.cfg.bottleneck_dim
        frame_size = self.cfg.sampled_frame
        #print(bottleneck)
        return {
            'CEL': torch.nn.CrossEntropyLoss(),
            'BCE': DomainBCE(),
            'HL': HLoss(),
            # trainable networks
            'F_e': FeatureEncoder(RESNET_DIM, bottleneck),
            'F_s': ResidualDialConvBlock(bottleneck, [1, 3, 7, 15], self.coeff),
            'D': DiscrminiatorBlock(bottleneck, [51, 45, 31, 1]),
            #'TRN': RelationModuleMultiScale(bottleneck, frame_size, self.cls_num),
            #'R': RelationModule(bottleneck, frame_size, self.cls_num),
            'C': Classifier(bottleneck, self.cls_num)
            #'self_loss': self_loss(),
        }

    def train_process(self, data, ctx):

        source = data['source']
        sfeat = source[:self.sample_number]
        strg = source[self.sample_number]
        target = data['target']
        tfeat = target[:self.sample_number]

        sfeat_0 = [self.F_e(i) for i in sfeat]
        tfeat_0 = [self.F_e(i) for i in tfeat]

        scls_logit = [self.C(i)[1] for i in sfeat_0]
        tcls_logit = [self.C(i)[1] for i in tfeat_0]
        L_cls = sum([self.CEL(i, strg) for i in scls_logit])
        L_ent = sum([self.HL(i) for i in tcls_logit])

        L_adv = []
        sfeat_0 = [ i.transpose(1, 2).contiguous() for i in sfeat_0]
        tfeat_0 = [ i.transpose(1, 2).contiguous() for i in tfeat_0]
        sfeat_group = [self.F_s(i) for i in sfeat_0]
        tfeat_group = [self.F_s(i) for i in tfeat_0]
        for i in range(self.sample_number):
            #sfeat_group[i].insert(0, sfeat_0[i])
            #tfeat_group[i].insert(0, tfeat_0[i])
            sdom_logit, sscal_w = self.D(sfeat_group[i])
            tdom_logit, tscal_w = self.D(tfeat_group[i])

            sdom_losses = torch.cat([self.BCE(l, 'S') for i, l in enumerate(sdom_logit)], dim=-1)
            tdom_losses = torch.cat([self.BCE(l, 'T') for i, l in enumerate(tdom_logit)], dim=-1)

            sdom_losses = torch.sum(sdom_losses * sscal_w, dim=-1)
            tdom_losses = torch.sum(tdom_losses * tscal_w, dim=-1)

            sdom_loss = torch.mean(sdom_losses)
            tdom_loss = torch.mean(tdom_losses)
            # + tdom_loss  L_ent +
            L_adv.append((sdom_loss + tdom_loss) / 2)
            #L_adv.append(tdom_loss)
        L_adv = sum(L_adv)

        L = (L_cls + L_ent + L_adv)/self.sample_number
        with self.optimize_config(
                optimer=pAdam(lr=self.cfg.lr, weight_decay=0.0005),
                # optimer=pSGD(lr=self.cfg.lr, momentum=0.9, weight_decay=0.0005, nesterov=True),
                lr_scheduler=pStepLR(step_size=self.cfg.lr_decay_epoch, gamma=self.cfg.lr_gamma),
        ):
            self.optimize_loss('global_loss', L, ['F_e', 'C', 'D', 'F_s'])


            # record metrics

    def eval_process(self, data, ctx, number=""):
        feats, trg = data[0:self.sample_number ], data[self.sample_number ]
        feats = [self.F_e(feat) for feat in feats]
        logits = [self.C(i) for i in feats]
        pred = logits[0][1]
        feats = logits[0][0]
        for i in range(1, self.sample_number):
            pred += logits[i][1]
            feats += logits[i][0]
        feats /= self.sample_number
        pred /= self.sample_number
        #self.fig_tsne(pred, trg, str(number))
        return pred, trg

    def coeff(self):
        alpha = 10
        high = self.cfg.adv_coeff
        low = 0
        bias = 0
        epoch = max(self.current_epoch - bias, 0)
        epoch = int(epoch)
        p = epoch / (self.cfg.epoch - bias)
        return np.float(2.0 * (high - low) / (1.0 + np.exp(-alpha * p)) - (high - low) + low)


def H(x):
    x_ = 1 - x
    return -(x * torch.log(x) + x_ * torch.log(x_))

def JSDivLoss(x, y):
    #print(x)
    M = (F.softmax(x, dim=1) + F.softmax(y, dim=1))/2
    b = M * F.log_softmax(x, dim=1) + M * F.log_softmax(y, dim=1)
    b = -b
    return b

