import torch
import torch.nn as nn
import torch.nn.functional as F
from math import factorial
import numpy as np
from sklearn.manifold import TSNE


def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1 or classname.find('ConvTranspose2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)
from torch.autograd import Function
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class CNNModel(nn.Module):

    def __init__(self):
        super(CNNModel, self).__init__()
        self.feature = nn.Sequential()
        self.feature.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature.add_module('f_relu1', nn.ReLU(True))
        self.feature.add_module('f_conv2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature.add_module('f_drop1', nn.Dropout2d())
        self.feature.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature.add_module('f_relu2', nn.ReLU(True))

        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50 * 4 * 4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_drop1', nn.Dropout2d())
        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm1d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))
        self.class_classifier.add_module('c_fc3', nn.Linear(100, 10))
        self.class_classifier.add_module('c_softmax', nn.LogSoftmax())

        self.domain_classifier = nn.Sequential()
        self.domain_classifier.add_module('d_fc1', nn.Linear(50 * 4 * 4, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm1d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))
        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 2))
        self.domain_classifier.add_module('d_softmax', nn.LogSoftmax(dim=1))
        self.apply(init_weights)

    def forward(self, input_data, alpha):
        input_data = input_data.expand(input_data.data.shape[0], 3, 28, 28)
        feature = self.feature(input_data)
        feature = feature.view(-1, 50 * 4 * 4)
        reverse_feature = ReverseLayerF.apply(feature, alpha)
        class_output = self.class_classifier(feature)
        domain_output = self.domain_classifier(reverse_feature)

        return class_output, domain_output

#特征提取
class FeatureEncoder(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.out_dim = out_dim
        linear = nn.Linear(in_dim, out_dim)
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=True),
            nn.ReLU(True),
        )
        self.apply(init_weights)

    def forward(self, feats: torch.Tensor):
        N, T, C = feats.shape
        #feats = feats.contiguous()
        #print(C)
        feats = feats.view(-1, C)
        feats = self.encoder(feats)
        feats = feats.view(N, T, self.out_dim)

        return feats


class Classifier(nn.Module):
    def __init__(self, in_dim, cls_num):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.classifer = nn.Linear(in_dim, cls_num)
        self.apply(init_weights)

    def forward(self, feats: torch.Tensor):
        feats = feats.transpose(1, 2).contiguous()
        B = feats.shape[0]
        feats = self.pool(feats).view(B, -1)
        logit = self.classifer(feats)
        return feats, logit

class DiscrminiatorBlock(nn.Module):
    def __init__(self, in_dim, seg_sizes):
        super().__init__()
        self.seg_sizes = seg_sizes
        self.discriminator_groups = nn.ModuleList(
            nn.ModuleList([Discriminator(in_dim) for _ in range(s)])
            for _, s in enumerate(seg_sizes))

        self.global_discriminators = nn.ModuleList(
            [NDiscriminator(in_dim) for _ in range(len(seg_sizes))])

    def forward(self, inputs):
        assert len(inputs) == len(self.discriminator_groups)

        logit_group = []
        for g_id, group in enumerate(self.discriminator_groups):
            inp = inputs[g_id]
            logits = []
            for d_id, dis in enumerate(group):
                feats = inp[:, :, d_id]
                logit = dis(feats)
                logits.append(logit)
            logits = torch.cat(logits, dim=1)
            logit_group.append(logits)


        scale_w = []
        for g_id, dis in enumerate(self.global_discriminators):
            inp = inputs[g_id]
            logit = dis(inp)
            pred = torch.sigmoid(logit)
            dis = torch.cat([pred, 1 - pred], dim=-1)
            #print(dis)
            ent = - 1.0 * torch.sum(torch.log(dis) * dis, dim=-1, keepdim=True)
            scale_w.append(ent.detach())

        scale_w = torch.cat(scale_w, dim=1)
        sum_scale = (torch.sum(scale_w, dim=1).view(len(scale_w), 1)).expand([-1, len(self.seg_sizes)])
        scale_w = scale_w / sum_scale
        return logit_group, scale_w


class Discriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 1024
        self.classifer = nn.Sequential(
            nn.Linear(in_dim, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, DIM),
            nn.ReLU(True),
            nn.Dropout(0.2),
            nn.Linear(DIM, 1),
        )
        self.apply(init_weights)

    def forward(self, feat):
        logit = self.classifer(feat)
        return logit

class NDiscriminator(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        DIM = 128
        from .gradient_reverse_layer import GradReverseLayer
        self.classifer = nn.Sequential(
            GradReverseLayer(lambda : 0),
            # nn.Linear(in_dim, DIM),
            # nn.ReLU(True),
            # nn.Linear(DIM, DIM),
            # nn.ReLU(True),
            nn.Linear(in_dim, 1),
        )
        self.apply(init_weights)


    def forward(self, feat):
        B, C, _ = feat.shape
        feat = F.adaptive_avg_pool1d(feat, 1)
        feat = feat.view(B, C)
        logit = self.classifer(feat)
        return logit


class ResidualDialConvBlock(nn.Module):
    def __init__(self, in_dim, dilations, coeff_fn=lambda: 1):
        super().__init__()
        from .gradient_reverse_layer import GradReverseLayer
        self.pre = nn.Sequential(
            GradReverseLayer(coeff_fn),
            # nn.Conv1d(in_dim, in_dim, kernel_size=1),
            )
        self.net = nn.Sequential(
            *[ResidualDialConv(in_dim, i) for i in dilations])

    def forward(self, inp, domain=None, w=None):
        #inp = inp.transpose(1, 2).contiguous()
        inp = self.pre(inp)
        itermidiate = []
        for layer in self.net:
            inp = layer(inp, domain, w)
            itermidiate.append(inp)
        return itermidiate


class ResidualDialConv(nn.Module):
    def __init__(self, in_dim, dilation, avg_pool=True):
        super().__init__()

        k_size = 3
        self.conv = nn.Sequential(
            nn.Conv1d(in_dim, in_dim, kernel_size=k_size, dilation=dilation),
            nn.ReLU(inplace=True), nn.Conv1d(in_dim, in_dim, kernel_size=1))
        self.un = nn.Unfold(kernel_size=[k_size, 1], dilation=dilation)
        self.k_size = k_size
        self.dilation = dilation

        self.apply(init_weights)

        if avg_pool:
            w = [1 / k_size ] * k_size
        else:
            w = list(range(k_size))
            mid = max(w) / 2
            w = [-1 * abs(i - mid) for i in w]
            w = np.exp(w) / sum(np.exp(w))
        w = torch.tensor(w).float()
        self.register_buffer('w', w)

    def forward(self, inp, domain=None, w=None):
        B, C, _ = inp.shape
        #print(inp.shape)
        T = self.k_size
        conv = self.conv(inp)  # shape [B, C, S]
        seg = self.un(inp.unsqueeze(-1)).view(B, C, T, -1)  # shape [B, C, T, S]
        #w = w.transpose(0, 2).contiguous()
        #print(seg.shape)
        if w == None:
            w = self.w.view(1, 1, -1, 1).expand_as(seg)
        else:
            temp = torch.zeros([seg.shape[0], seg.shape[2], seg.shape[3]]).cuda()
            #print(self.dilation)
            sig = nn.Sigmoid()
            w = sig(w[0])
            #print(w)
            #w = 1 / (w * (1 - w))
            if domain == 'T':
                w = 1 / (1 - w)
            else:
                w = 1 / w
            for i in range(seg.shape[0]):
                for k in range(seg.shape[3]):
                    now = w[i][k] + w[i][k + self.dilation] + w[i][k + self.dilation * 2]
                    for j in range(seg.shape[2]):
                        # print(w[i][k + j * self.dilation] / now)
                        temp[i][j][k] = w[i][k + j * self.dilation] / now
            w = temp.view(seg.shape[0], -1, seg.shape[2], seg.shape[3]).expand_as(seg)
        #print(w)
        residual = torch.sum(seg * w, dim=2)
        out = conv + residual
        #out = out[ :, :, ::3]
        #print("laji", out.shape)
        return out


# the relation consensus module by Bolei
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pdb


class RelationModuleMultiScale(torch.nn.Module):
    # Temporal Relation module in multiply scale, suming over [2-frame relation, 3-frame relation, ..., n-frame relation]

    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModuleMultiScale, self).__init__()
        self.subsample_num = 3 # how many relations selected to sum up
        self.img_feature_dim = img_feature_dim
        self.scales = [i for i in range(num_frames, 1, -1)] # generate the multiple frame relations

        self.relations_scales = []
        self.subsample_scales = []
        for scale in self.scales:
            relations_scale = self.return_relationset(num_frames, scale)
            self.relations_scales.append(relations_scale)
            self.subsample_scales.append(min(self.subsample_num, len(relations_scale))) # how many samples of relation to select in each forward pass

        self.num_class = num_class
        self.num_frames = num_frames
        num_bottleneck = 256
        self.fc_fusion_scales = nn.ModuleList() # high-tech modulelist
        for i in range(len(self.scales)):
            scale = self.scales[i]
            fc_fusion = nn.Sequential(
                        nn.ReLU(),
                        nn.Linear(scale * self.img_feature_dim, num_bottleneck),
                        nn.ReLU(),
                        nn.Linear(num_bottleneck, self.num_class),
                        )
            self.fc_fusion_scales += [fc_fusion]

        print('Multi-Scale Temporal Relation Network Module in use', ['%d-frame relation' % i for i in self.scales])

    def forward(self, input):
        # the first one is the largest scale
        act_all = input[:, self.relations_scales[0][0] , :]
        act_all = act_all.view(act_all.size(0), self.scales[0] * self.img_feature_dim)
        act_all = self.fc_fusion_scales[0](act_all)

        for scaleID in range(1, len(self.scales)):
            # iterate over the scales
            idx_relations_randomsample = np.random.choice(len(self.relations_scales[scaleID]), self.subsample_scales[scaleID], replace=False)
            for idx in idx_relations_randomsample:
                act_relation = input[:, self.relations_scales[scaleID][idx], :]
                act_relation = act_relation.view(act_relation.size(0), self.scales[scaleID] * self.img_feature_dim)
                act_relation = self.fc_fusion_scales[scaleID](act_relation)
                act_all += act_relation
        return act_all

    def return_relationset(self, num_frames, num_frames_relation):
        import itertools
        return list(itertools.combinations([i for i in range(num_frames)], num_frames_relation))

class RelationModule(torch.nn.Module):
    # this is the naive implementation of the n-frame relation module, as num_frames == num_frames_relation
    def __init__(self, img_feature_dim, num_frames, num_class):
        super(RelationModule, self).__init__()
        self.num_frames = num_frames
        self.num_class = num_class
        self.img_feature_dim = img_feature_dim
        self.classifier = self.fc_fusion()
    def fc_fusion(self):
        # naive concatenate
        num_bottleneck = 512
        classifier = nn.Sequential(
                nn.ReLU(),
                nn.Linear(self.num_frames * self.img_feature_dim, num_bottleneck),
                nn.ReLU(),
                nn.Linear(num_bottleneck,self.num_class),
                )
        return classifier
    def forward(self, input):
        input = input.view(input.size(0), self.num_frames*self.img_feature_dim)
        input = self.classifier(input)
        return input

if __name__ == "__main__":
    batch_size = 32
    num_frames = 9
    num_class = 12
    img_feature_dim = 512
    input_var = Variable(torch.randn(batch_size, num_frames, img_feature_dim))
    print("test 1")
    model = RelationModule(img_feature_dim, num_frames, num_class)
    output = model(input_var)
    print(output)
    print(output.shape)
