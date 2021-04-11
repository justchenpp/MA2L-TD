import torch
from torch import nn
import numpy as np
from sklearn.cluster import k_means
import torch.nn.functional as F

class TemporalSlidingSegment(nn.Module):
    def __init__(self, size, stride=None):
        super().__init__()
        stride = size if stride is None else stride
        self.un = nn.Unfold(kernel_size=[size, 1], stride=stride)
        self.size = size

    def forward(self, feat:torch.Tensor):
        B, _, C = feat.shape
        feat = feat.transpose(2, 1).unsqueeze(-1)  # shape [B, C, T, 1]      
        unfold_tfeat = self.un(feat)  # shape [B, C*window_size, slided_size]
        unfold_tfeat = unfold_tfeat.view(B, C, self.size, -1) 
        slided_feat = unfold_tfeat.transpose(1, 3)  # shape [B, slided_size, window_size, C]
        return slided_feat.contiguous()


class SegmentPooling(nn.Module):
    def __init__(self, size):
        super().__init__()
        w = list(range(size))
        mid = max(w)/2
        w = [-1 * abs(i - mid) for i in w]
        w = [[np.exp(w)/sum(np.exp(w))]]
        w = torch.tensor(w).float()
        self.register_buffer('w', w)
    
    def forward(self, feats: torch.Tensor):
        B, S, T, C = feats.shape
        feats = feats.view(B*S, T, C)
        w = self.w.expand(B*S, -1, -1)
        seg_pooled = torch.bmm(w, feats).squeeze()
        seg_pooled = seg_pooled.view(B, S, C)       
        return seg_pooled

class SegmentCluster(nn.Module):
    def __init__(self, cluster_num, kmean=False):
        super().__init__()
        self.cluster_num = cluster_num
        self.kmean = kmean
    
    def forward(self, feat, win_feats, seg_feats):
        B, S, T, C = win_feats.shape
        if self.kmean:
            arrarys = feat.detach().cpu().numpy()
            cets = list()
            for i in range(len(arrarys)):
                arrary = arrarys[i]
                cet, _, _ = k_means(arrary, self.cluster_num)
                cets.append(cet)
            cets = np.array(cets)
            cets = torch.from_numpy(cets).to(feat.device)

            shape = [B, self.cluster_num, S, C]
            sims = F.cosine_similarity(
                seg_feats.unsqueeze(1).expand(shape),
                cets.unsqueeze(2).expand(shape),
                dim = -1
            )
            _, selected_seg_idxs = torch.max(sims, dim=-1)
            selected_seg_idxs, _ = selected_seg_idxs.sort(dim=-1)
        else:
            step = int(S/3)
            idxs = list(range(S))[1::step]
            selected_seg_idxs = torch.from_numpy(np.array([idxs]*B)).cuda()
        
        _win_feats = list()
        _seg_feats = list()
        for b in range(B):
            idx = selected_seg_idxs[b]
            win_feat = win_feats[b]
            seg_feat = seg_feats[b]
            _win_feats.append(win_feat.index_select(0, idx).unsqueeze(0))
            _seg_feats.append(seg_feat.index_select(0, idx).unsqueeze(0))
            # x, c = torch.unique(idx, return_counts=True)
            # print(len(c))
        win_feats = torch.cat(_win_feats, dim=0)
        seg_feats = torch.cat(_seg_feats, dim=0)

        uni = [len(torch.unique(i)) for i in selected_seg_idxs]
        avg_uni = sum(uni)/len(uni)
        self.avg_uni = avg_uni

        return win_feats, seg_feats