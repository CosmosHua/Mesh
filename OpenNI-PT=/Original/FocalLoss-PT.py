# coding:utf-8
# !/usr/bin/python3

import torch
import torch.nn as nn


################################################################################
class FocalLoss(nn.Module):
    """Loss(X,k) = -weight(k) * (1-softmax(X,k))^gamma * log(softmax(X,k)).
    Default: The losses are averaged across observations for each minibatch.
    # Ref: https://arxiv.org/abs/1708.02002
    Args:
        weight(1D tensor|class_num): the weighting factor for class imbalance.
        gamma(float>0): reduces the relative loss for well-classiﬁed examples(p>0.5),
            in order to put more focus on hard, misclassiﬁed examples. Default: 2.
        reduction(string): the reduction to apply to the output: 'none'|'mean'|'sum'.
            'none': no reduction will be applied; 'sum': the output will be summed;
            'mean': the mean value of the outputs. Default: 'mean'."""
    def __init__(self, weight, gamma=2, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.gamma = gamma; self.rd = reduction
        assert isinstance(weight, (int, torch.Tensor))
        self.wt = torch.ones(weight) if type(weight)==int else weight

    def forward(self, input, target):
        """Args: input (2D Tensor): (N=batch_size, C=class_num), probs.
                 target (1D Tensor): (N=batch_size), labels.
        Returns: the focal loss of input/inference and taget/label."""
        wt = self.wt.to(input.device)
        p = torch.softmax(input, dim=1)
        loss = -wt * (1-p)**self.gamma * p.log()
        
        # Method_1:
        mask = torch.zeros_like(loss)
        mask.scatter_(1, target.view(-1,1), 1.0) # dim,index,src
        loss = (loss*mask).sum(dim=1)
        # Method_2:
        #loss = [loss[i,k] for i,k in enumerate(target)]
        #loss = torch.tensor(loss, device=wt.device)
        
        if self.rd=="mean": loss = loss.mean()
        elif self.rd=="sum": loss = loss.sum()
        return loss # also for "none"


################################################################################
if __name__ == "__main__":
    N, C = 4, 7 # batch, class_num
    x = torch.rand((N,C)) # (N,C)
    y = torch.randint(0,C,(N,)) # (N)
    wt = torch.rand((C,)); #print(wt)
    
    rd = ("none", "sum", "mean")
    # CE(X,k) = -wt(k)*log(softmax(x,k))
    CE = [nn.CrossEntropyLoss(wt, reduction=i) for i in rd]
    # FL(X,k) = -wt(k)*log(softmax(x,k))*(1-softmax(X,k))^gamma
    FL = [FocalLoss(wt, gamma=0, reduction=i) for i in rd]
    
    CE = [f(x,y) for f in CE]; print("CE =", CE)
    FL = [f(x,y) for f in FL]; print("FL =", FL)
