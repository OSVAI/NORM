from __future__ import print_function

import torch.nn as nn


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import ipdb


class EMDLoss(nn.Module):
    def __init__(self, reduction='none'):
        super(EMDLoss, self).__init__()
        self.reduction = reduction
        self.sinkhorn = SinkhornDistance()

    def get_batchcost(self, a, b):
        Ba, N, Da = a.shape
        Bb, M, Db = b.shape
        assert(Ba == Bb), "Batch size unmatched!"
        assert(Da == Db), "vector dimension unmatched!"

        a = F.normalize(a, dim=2)
        b = F.normalize(b, dim=2)

        cost = 1 - (torch.bmm(a, b.permute(0,2,1)))
        return cost

    def get_batchweights(self, cost):
        S = 1-cost

        #margin_a, _ = S.max(dim=2)
        #margin_b, _ = S.max(dim=1)
        margin_a = S.mean(dim=2)
        margin_b = S.mean(dim=1)

        #margin_a = F.softmax(margin_a,  dim=1)
        #margin_b = F.softmax(margin_b,  dim=1)
        margin_a = F.normalize(margin_a, p=1, dim=1).clamp_min(0)
        margin_b = F.normalize(margin_b, p=1, dim=1).clamp_min(0)

        margin_a = F.normalize(margin_a, p=1, dim=1)
        margin_b = F.normalize(margin_b, p=1, dim=1)

        return margin_a, margin_b

    def forward(self, a, b):
        cost = self.get_batchcost(a,b)
        wa, wb = self.get_batchweights(cost.detach())
        D = self.sinkhorn(wa, wb, cost)
        #ipdb.set_trace()
        if torch.isnan(D.sum()):
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~NAN!!!~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            D.fill_(1)
        return D


class SinkhornDistance(torch.nn.Module):
    r"""
        Given two empirical measures each with :math:`P_1` locations
        :math:`x\in\mathbb{R}^{D_1}` and :math:`P_2` locations :math:`y\in\mathbb{R}^{D_2}`,
        outputs an approximation of the regularized OT cost for point clouds.
        Args:
        eps (float): regularization coefficient
        max_iter (int): maximum number of Sinkhorn iterations
        reduction (string, optional): Specifies the reduction to apply to the output:
        'none' | 'mean' | 'sum'. 'none': no reduction will be applied,
        'mean': the sum of the output will be divided by the number of
        elements in the output, 'sum': the output will be summed. Default: 'none'
        Shape:
            - Input: :math:`(N, P_1, D_1)`, :math:`(N, P_2, D_2)`
            - Output: :math:`(N)` or :math:`()`, depending on `reduction`
    """

    def __init__(self, eps=1e-3, max_iter=10, reduction='none'):
        super(SinkhornDistance, self).__init__()
        self.eps = eps
        self.max_iter = max_iter
        self.reduction = reduction

    def forward(self, mu, nu, C):
        u = torch.zeros_like(mu)
        v = torch.zeros_like(nu)
        # To check if algorithm terminates because of threshold
        # or max iterations reached
        actual_nits = 0
        # Stopping criterion
        thresh = 1e-1

        # Sinkhorn iterations
        for i in range(self.max_iter):
            u1 = u  # useful to check the update
            u = self.eps * \
                (torch.log(
                    mu+1e-8) - torch.logsumexp(self.M(C, u, v), dim=-1)) + u
            v = self.eps * \
                (torch.log(
                    nu+1e-8) - torch.logsumexp(self.M(C, u, v).transpose(-2, -1), dim=-1)) + v
            err = (
                u - u1).abs().sum(-1).mean()

            actual_nits += 1
            if err.item() < thresh:
                break

        U, V = u, v
        # Transport plan pi = diag(a)*K*diag(b)
        pi = torch.exp(
            self.M(C, U, V)).detach()
        # Sinkhorn distance
        cost = torch.sum(
            pi * (C), dim=(-2, -1))
        self.actual_nits = actual_nits
        if self.reduction == 'mean':
            cost = cost.mean()
        elif self.reduction == 'sum':
            cost = cost.sum()

        #return cost, pi, C
        return cost

    def M(self, C, u, v):
        '''
        "Modified cost for logarithmic updates"
        "$M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$"
        '''
        return (-C + u.unsqueeze(-1) + v.unsqueeze(-2)) / self.eps

def dropout(X, Y, drop_prob):
    X = X.float()
    assert 0<=drop_prob<=1
    keep_prob = 1-drop_prob
    if keep_prob==0:
        return torch.torch.zeros_like(X)
    mask = (torch.rand(X.shape)<keep_prob).float()
    mask = mask.cuda()

    return mask*X/keep_prob, mask*Y/keep_prob


class HintLoss2(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(HintLoss, self).__init__()
		self.criterion = EMDLoss()
	def forward(self, fm_s, fm_t):
		b, c, h, w = fm_t.shape
		fm_s = fm_s.permute(0,2,3,1).reshape(b, h*w, c)
		fm_t = fm_t.permute(0,2,3,1).reshape(b, h*w, c)

		loss = self.criterion(fm_s, fm_t).mean()
		#loss = F.mse_loss(fm_s, fm_t)
		return loss

class HintLoss(nn.Module):
	'''
	FitNets: Hints for Thin Deep Nets
	https://arxiv.org/pdf/1412.6550.pdf
	'''
	def __init__(self):
		super(HintLoss, self).__init__()
		self.criterion = EMDLoss()
	def forward(self, fm_s, fm_t):
		loss = F.mse_loss(fm_s, fm_t)
		return loss


class HintLoss1(nn.Module):
    """Fitnets: hints for thin deep nets, ICLR 2015"""
    def __init__(self):
        super(HintLoss1, self).__init__()
        self.crit = nn.MSELoss()

    def forward(self, f_s, f_t):
        loss = self.crit(f_s, f_t)
        return loss
