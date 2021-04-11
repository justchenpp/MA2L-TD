import torch
import torch.nn as nn
import torch.nn.functional as F

#交叉熵
class DCrossEntropyLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, logit, targ):
        assert logit.shape == targ.shape
        pred = F.log_softmax(logit, dim=1)
        losses = torch.sum(-targ * pred, 1)
        loss = torch.mean(losses)
        return loss

class self_loss(nn.Module):
    def __init__(self, num, label_num):
        self.num = num
        self.label_num = label_num
        super().__init__()

    def forward(self, logits):
        ans = []
        max_label = [torch.argmax(logit, dim=1) for logit in logits]
        pred = torch.zeros([logits[0].shape[0], self.label_num]).cuda()
        for i in range(logits[0].shape[0]):
            temp = torch.zeros([self.label_num]).cuda()
            temp_max, temp_x = 0, 0
            for j in range(self.num):
                temp[int(max_label[j][i])] += 1
                if temp[int(max_label[j][i])] > temp_max:
                    temp_x = int(max_label[j][i])
                    temp_max = temp[int(max_label[j][i])]
            pred[i][temp_x] += 1
        pred = torch.argmax(pred, dim=1)
        for i in range(self.num):
            Loss = torch.nn.CrossEntropyLoss()
            ans.append(Loss(logits[i], pred))
        return sum(ans)

#实现域判别
class DomainBCE(nn.Module):
    def __init__(self):
        super().__init__()
        self.BCE = nn.BCEWithLogitsLoss(reduction='none')
        self.register_buffer('S', torch.zeros(1,1))
        self.register_buffer('T', torch.ones(1,1))
    
    def forward(self, logit, domain):
        B, T = logit.shape
        logit = logit.view(B*T, 1)
        if domain == 'S':
            trg = torch.zeros_like(logit)
        else:
            trg = torch.ones_like(logit)
        loss = self.BCE(logit, trg)
        loss = loss.view(B, T)
        loss = torch.mean(loss, dim=-1, keepdim=True)
        return loss

#熵
class HLoss(torch.nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.mean()
        return b
