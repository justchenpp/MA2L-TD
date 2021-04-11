import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import permutations, product


class SSSignalGenerator(nn.Module):
    def __init__(self, cluster_num):
        super().__init__()
        self.cluster_num = cluster_num
        tmp_order = list(permutations(range(cluster_num), cluster_num))
        dom_order = list(product(*[[0, 1]] * cluster_num))
        self.register_buffer('tmp_order_set', torch.from_numpy(np.array(tmp_order)))
        self.register_buffer('dom_order_set', torch.from_numpy(np.array(dom_order)))
        self.register_buffer('dom_order_bias', torch.arange(0, cluster_num))
        self.register_buffer('dom_confuse_lab', torch.ones(1, len(dom_order)) / len(dom_order))
        self.register_buffer('tmp_confuse_lab', torch.ones(1, len(tmp_order)) / len(tmp_order))

    def forward(self, sfeat, tfeat):
        B = sfeat.shape[0]
        D = sfeat.device
        # randomly generate temporal orders
        tem_rand_lab = torch.randint(0, self.tmp_order_set.shape[0], (B, ), device=D)
        tmp_orders = self.tmp_order_set.index_select(0, tem_rand_lab)
        # sfeat = self._batch_order_select(sfeat, tmp_orders)
        # tfeat = self._batch_order_select(tfeat, tmp_orders)
        tmp_lab = torch.cat([tem_rand_lab, tem_rand_lab], dim=0)

        # randomly gnerate domain orders
        dom_order_len = self.dom_order_set.shape[0]
        dom_rand_lab1 = torch.randint(0, int(dom_order_len / 2), (B, ), device=D)
        dom_rand_lab2 = dom_order_len - 1 - dom_rand_lab1
        # dom_rand_idx = torch.cat([dom_rand_idx, c_dom_rand_idx], dim=0)
        dom_orders1 = self.dom_order_set.index_select(0, dom_rand_lab1)
        dom_orders2 = self.dom_order_set.index_select(0, dom_rand_lab2)

        feat1 = self.dom_order_select(sfeat, tfeat, dom_orders1)
        feat2 = self.dom_order_select(sfeat, tfeat, dom_orders2)
        feat = torch.cat([feat1, feat2], dim=0)
        dom_lab = torch.cat([dom_rand_lab1, dom_rand_lab2], dim=0)


        dom_conf_lab = self.dom_confuse_lab.expand(B*2 , -1)
        tmp_conf_lab = self.tmp_confuse_lab.expand(B*2 , -1)
 
        return feat, (dom_lab, dom_conf_lab), (tmp_lab, tmp_conf_lab)

    def _batch_order_select(self, batch, orders, dim=0):
        result = list()
        for i in range(len(batch)):
            t = batch[i].index_select(dim, orders[i])
            result.append(t.unsqueeze(0))
        return torch.cat(result, dim=0)

    def dom_order_select(self, sfeat, tfeat, orders):
        feat = torch.cat([sfeat, tfeat], dim=1)
        step = self.cluster_num - 1
        orders = orders + self.dom_order_bias + (orders == 1).long() * step
        return self._batch_order_select(feat, orders)


if __name__ == "__main__":
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    M = SSSignalGenerator(3).cuda()
    s = torch.rand(1, 3, 1)
    t = torch.rand(1, 3, 1)
    ss,dl,tl = M(s, t)
    print(s)
    print(t)
    print('---------------------')
    print(ss)
    print('---------------------')
    print(dl[0])
    print('---------------------')
    print(M.dom_order_set)
    print('---------------------')
    print(tl[0])
    print('---------------------')
    print(M.tmp_order_set)

    