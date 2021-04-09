import torch
import torch.nn

class ContrastiveLoss(nn.Module):

    def __init__(self, temp=0.5, normalize= False):
        super().__init__()
        self.temp = temp
        self.normalize = normalize

    def forward(self,xi,xj):

        x = torch.cat((xi,xj),dim=0)
        is_cuda = xi.is_cuda 

        # nominator : -->  e^ (sim(positive pair) / temp)
        # why normalize ?
        if self.normalize:
            sim_match_denom = torch.norm(xi, dim=1) * torch.norm(xj, dim=1)
            sim_match = torch.exp((torch.sum(xi * xj, dim = -1)/sim_match_denom) / self.temp) 
        else: 
            sim_match = torch.exp(torch.sum(xi * xj, dim = -1)/ self.temp)

        # mutual similarities between each paris
        sim_mat = torch.mm(x,x.T)
        if self.normalize:
            # x*x.T
            sim_mat_norm = torch.mm(torch.norm(x, dim=1).unsqueeze(1),  torch.norm(x, dim=1).unsqueeze(1).T )
            sim_mat = sim_mat / sim_mat_norm.clamp(min=1e-16)
        else: 
            sim_mat = sim_match = torch.exp(torch.sum(xi * xj, dim=-1) / self.temp)

        norm_sum = torch.exp(torch.ones(x.size(0)) / self.temp)
        norm_sum = norm_sum.cuda() if is_cuda else norm_sum
        loss = torch.mean(-torch.log(sim_match/ (torch.sum(sim_mat, dim=-1)- norm_sum)))
        return loss

