import clip
import faiss
import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils import *
import data.data_config as config
from scipy.stats import rankdata
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap

def cal_topk_acc(logits, category_labels, k = 1):
    clr_num, topk_num = logits.shape[-1], logits.shape[-1] * [False]
    topk = torch.sort(logits.softmax(dim = -1), -1, descending = True)[1][:, :k].cpu()
    #top1 = torch.max(logits.softmax(dim = -1), -1)[1].cpu()
    category_labels = [torch.tensor(range(clr_num))[category_labels[i].bool()] for i in range(clr_num)]
    for j in range(k):
        temp_acc = [topk[i][j] in category_labels[i] for i in range(clr_num)]
        topk_num = [temp_acc[i] or topk_num[i] for i in range(clr_num)]
    return np.sum(topk_num) / clr_num

def cal_matching_matrix(fmri_categories, img_categories, DEVICE):
    img_categories = np.array(img_categories)
    fmri_categories = np.array(fmri_categories)
    matching_labels = np.array([fmri_categories[i] == img_categories for i in range(len(fmri_categories))])
    img_dustbin = ~np.any(matching_labels, 0)
    fmri_dustbin = np.hstack((~np.any(matching_labels, 1), False))[:,None]
    aug_matching_labels = np.vstack((matching_labels, img_dustbin))
    aug_matching_labels = np.hstack((aug_matching_labels, fmri_dustbin))
    return torch.tensor(matching_labels).to(DEVICE), torch.tensor(aug_matching_labels).to(DEVICE)