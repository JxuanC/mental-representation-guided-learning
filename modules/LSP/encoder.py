import torch
import logging
import numpy as np
from math import log
from torch import nn
import torch.utils.data
from einops import rearrange
from torch.autograd import Function
from torch.nn import functional as F
import random
from modules.LSP.vit import fMRI_ViT_Encoder
from modules.LSP.gnn import AttentionalGNN
from modules.LSP.sampler import GumbelSoftmaxSampler
from modules.LSP.GW_OT import GW_distance_uniform
from modules.LSP.optimal_Transport import log_optimal_transport

class SimpleMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # 输入层到隐藏层
        self.relu = nn.ReLU()  # ReLU激活函数
        self.fc2 = nn.Linear(hidden_size, output_size)  # 隐藏层到输出层

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

class lsp(nn.Module):
    def __init__(self, fmri_dim, rois_len, topk, embed_dim, depth, num_heads, clip_features, retrieval_index, visual_dim = 512):
        super(lsp, self).__init__()
        self.clip_features = clip_features
        self.retrieval_index = retrieval_index
        self.topk = topk
        self.sampler = GumbelSoftmaxSampler()
        self.fmri_encoder = fMRI_ViT_Encoder(512, rois_len, embed_dim, depth, num_heads)
        self.img_encoder = SimpleMLP(visual_dim, 256, visual_dim)
        self.retrieval_encoder = fMRI_ViT_Encoder(embed_dim, topk + 1, embed_dim, depth, num_heads)
        self.gnn = AttentionalGNN(visual_dim, ['self', 'self', 'self'])
        self.final_proj = nn.Conv1d(512, 512, kernel_size = 1, bias = True)
        self.start_proj = nn.Linear(fmri_dim, 512)
    
    def retrieval(self, fmri):
        D, I = self.retrieval_index.search(fmri.detach().cpu().numpy(), self.topk)
        return self.clip_features[I]
    
    def interpolation(self, x, noise = False):
        if(noise):
            x = x + torch.randn_like(x).to(x)
        fmri_num = x.shape[1]
        selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
        if(selected_no.shape[0] != 1):
            coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0]), dtype = torch.float32).softmax(0)
            mixup_x = torch.sum(x[:, selected_no] * coefficient[None, :, None, None].to(x.device), 1)
            return mixup_x
        return x[:, selected_no, :, :].squeeze()
    
    def neighbor_sampling(self, x, neighbor_size):
        #x = x / x.norm(dim=-1, keepdim=True)
        self_similarity = torch.matmul(x, x.T)
        ont_hot_hard, ont_hot_soft = self.sampler.sampling(self_similarity, neighbor_size)
        return ont_hot_hard, ont_hot_soft

    @torch.no_grad()
    def neighbor_label(self, x_hard, y_hard, neighbor_size):
        neighbor_label = (torch.nonzero(x_hard)[:,1] == torch.nonzero(y_hard)[:,1])
        if neighbor_label.shape[0] % neighbor_size != 0:
            raise ValueError(f"x_hard shape is {x_hard.shape}, y_hard shape is {y_hard.shape}, neighbor_label shape is {neighbor_label.shape}, neighbor_size is {neighbor_size}")
        neighbor_label = neighbor_label.reshape(-1, neighbor_size)
        return torch.bmm(neighbor_label[:,:,None].float(), neighbor_label[:,None,:].float())

    def LoSC_loss(self, x, y, neighbor_size):
        x = x / x.norm(dim=-1, keepdim=True)
        y = y / y.norm(dim=-1, keepdim=True)
        plan_soft = self.W_OT_loss(torch.matmul(x, y.T))[0]
        plan_hard = F.gumbel_softmax(plan_soft, tau=0.01, hard=True)
        x_hard, x_soft = self.neighbor_sampling(x, neighbor_size)
        y_hard, y_soft = self.neighbor_sampling(y, neighbor_size)
        neighbor_x = x.repeat(x.shape[0],1,1) * x_hard.unsqueeze(-1)
        neighbor_x = neighbor_x[x_hard != 0].view(x_hard.shape[0], -1, x.shape[-1])
        neighbor_y = y.repeat(y.shape[0],1,1) * y_hard.unsqueeze(-1)
        neighbor_y = neighbor_y[y_hard != 0].view(y_hard.shape[0], -1, y.shape[-1])
        neighbor_yt = torch.matmul(plan_hard, neighbor_y.reshape(neighbor_x.shape[0], -1)).reshape(neighbor_x.shape)
        neighbor_label = self.neighbor_label(x_hard, y_hard, neighbor_size)
        gwd, P, C = self.GW_OT_loss(neighbor_x, neighbor_yt, neighbor_label)
        return gwd, P, C, x_soft, y_soft, neighbor_label
        
    def GW_OT_loss(self, x, y, indices = None):
        gwd, P, C = GW_distance_uniform(x.transpose(2,1), y.transpose(2,1), 
                                        lamda = 0.02, iteration = 100, OT_iteration = 100, indices = indices)
        return gwd, P, C
    
    def W_OT_loss(self, cost_matrix):
        scores = log_optimal_transport(cost_matrix.unsqueeze(0), 
                                    torch.nn.Parameter(torch.tensor(1.0)).to(cost_matrix), iters = 100)
        return scores

    def forward(self, fmri, img):
        # x (batch, roi_num, roi_dim)
        x = self.start_proj(fmri)
        if(len(x.shape) == 4):
            x = self.interpolation(x)
        y = self.img_encoder(img)
        #y = self.fmri_encoder(y.view_as(x))
        x = self.fmri_encoder(x)
        x, y = self.gnn(rearrange(x, '(b n) d -> b d n', b = 1), rearrange(y, '(b n) d -> b d n', b = 1))
        x, y = rearrange(x, 'b d n -> (b n) d', b = 1), rearrange(y, 'b d n -> (b n) d', b = 1)
        if(self.topk > 0):
            y = self.retrieval(x)
            y = torch.tensor(y, dtype = torch.float32).to(x.device)
            x, y = self.gnn(rearrange(x, '(b n) d -> b d n', b = 1), rearrange(y, 'n k d -> k d n'))
            return rearrange(self.final_proj(x), 'b d n -> (b n) d')
        else:
            return x.squeeze(), y.squeeze()
        
    def encode_fmri(self, fmri, return_class_embedding = True):
        x = self.start_proj(fmri)
        x = self.fmri_encoder(x, return_class_embedding)
        return x
    
    def encode_image(self, image, use_gnn = True):
        x = self.img_encoder(image)
        if(use_gnn):
            x = self.gnn.self_embedding(rearrange(x, '(b n) d -> b d n', b = 1))
            x = rearrange(x, 'b d n -> (b n) d', b = 1)
        return x.squeeze()