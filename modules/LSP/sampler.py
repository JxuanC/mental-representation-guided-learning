import torch

class GumbelSoftmaxSampler():
    """Sample based on a Gumbel-Max distribution.

    Use re-param trick for back-prop
    """
    def __init__(self, tau=1., device='cuda'):
        #self.num_samples = num_samples
        # self.num_points = num_points
        self.device = device
        self.gumbel_dist = torch.distributions.gumbel.Gumbel(
                torch.tensor(0.),
                torch.tensor(1.))
        self.tau = tau

    def sampling(self, logits, num_samples, selected = None):
        if selected is None:
            gumbels = self.gumbel_dist.sample(logits.shape).to(logits)
            gumbels = (logits + gumbels)/self.tau
            y_soft = gumbels.softmax(-1)
            topk = torch.topk(gumbels, num_samples, dim=-1)
            y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(-1, topk.indices, 1.0)
            ret = y_hard - y_soft.detach() + y_soft
        else:
            pass

        return ret, y_soft#, topk.indices