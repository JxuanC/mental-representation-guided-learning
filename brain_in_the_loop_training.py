import os
import clip
import faiss
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from modules.LSP.encoder import lsp
import data.data_config as config
from data.data_augment import get_visual_fmri_dataset
from utils import *
from data.DIR import DIR_sub_with_images
from modules.matching_loss import *
from modules.LSP.optimal_Transport import log_optimal_transport
from modules.matching_evaluation import *
import argparse

def epoch_train(model, optimizer, train_dl, save_path, epoch, DEVICE):
    running_loss, running_acc = [], []
    for idx, (image_clips, fmris, img_categories, fmri_categories) in enumerate(train_dl):
        model.train()
        image_clips = image_clips.float().to(DEVICE)
        fmris = fmris.float().to(DEVICE)
        
        fmri_embedding, virtual_embedding = model(fmris, image_clips)
        concat_embedding = torch.concat((fmri_embedding, virtual_embedding), 0)
        concat_embedding = concat_embedding / concat_embedding.norm(dim=-1, keepdim=True)
        fmri_embedding = concat_embedding[:fmri_embedding.shape[0]]
        img_embedding = image_clips / image_clips.norm(dim=-1, keepdim=True)
        cross_similarity = torch.matmul(concat_embedding, img_embedding.T)

        concat_categories = fmri_categories + img_categories
        across_labels, aug_across_labels = cal_matching_matrix(concat_categories, img_categories)
        self_labels, aug_self_labels = cal_matching_matrix(concat_categories, concat_categories)
        scores = log_optimal_transport(cross_similarity.unsqueeze(0), 
                                       torch.nn.Parameter(torch.tensor(1.0)).to(DEVICE), iters = 100)
        ntxent_loss, self_similarity = ntxent_loss_with_soft_labels(concat_embedding, concat_embedding, self_labels.float())
        loss = torch.mean(-scores[across_labels[None,:,:]]) + 0.1 * ntxent_loss# + 10 * model.LoSC_loss(fmri_embedding, img_embedding)
        gwd, P, C, x_soft, y_soft, neighbor_label = model.LoSC_loss(fmri_embedding, virtual_embedding, 20)
        loss = loss + F.cross_entropy(x_soft, y_soft) + 100 * torch.sum(gwd)/torch.log(torch.sum(neighbor_label)) + 10 * torch.mean(-P[neighbor_label.bool()])
        top1_acc = cal_topk_acc(scores[0,:,:], across_labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss.append(loss.item())
        running_acc.append(top1_acc.item())

        if idx % 10 == 0: # print every 10 mini-batches
            trained_num = idx * image_clips.shape[0]
            data_num = len(train_dl) * image_clips.shape[0]
            percent = int(100. * trained_num / data_num)
            logging.info(f"Epoch: {(epoch + 1):4d} Batch: {(idx + 1):4d} [{(trained_num):5d}/{(data_num):5d} ({(percent):2d}%)]" +
                  f"  Top1: {(np.mean(running_acc) * 100):.2f}% Loss: {(np.mean(running_loss)):.4f}")
            running_loss = []
            running_acc = []
    save_checkpoint(model, epoch + 1, idx + 1, save_path)

def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    proxy = args.proxy
    model_name = args.model_size
    rois = args.rois
    selected_rois, selected_sub = config.ROIS[rois], args.sub
    mixup, candidate, random_matching = True, True, False
    BATCH_SIZE = args.batch_size
    train_visual_features = h5py.File(f"./pretrained-feature/{proxy}/{model_name}/kamitani_train.hdf5", 'r')
    test_visual_features = h5py.File(f"./pretrained-feature/{proxy}/{model_name}/kamitani_test.hdf5", 'r')

    log_path = setup_logging_from_args(f'{config.SAVE_DIR}/LSP/{proxy}/{model_name}/{selected_sub}/{rois}/', 
                                        datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + f'_b{BATCH_SIZE}')
    vit_clip, preprocess = clip.load("ViT-L/14")#ViT-B/16,ViT-B/32,RN50,RN101
    train_dl, test_dl, fmri_dim = get_visual_fmri_dataset('DIR', selected_sub, selected_rois, 
                                                            [train_visual_features, test_visual_features], BATCH_SIZE, mixup, candidate, random_matching)

    _, _, _, testRois, testStis, testStiIDs = DIR_sub_with_images(preprocess, sub = selected_sub, rois = selected_rois) #fmri-image

    testStis = torch.tensor(np.vstack([test_visual_features[id.split('.')[0]][:] for id in testStiIDs])).to(DEVICE)
    brain_supervision = lsp(fmri_dim, len(selected_rois), 0, testStis.shape[-1], 12, 8, None, None, visual_dim = testStis.shape[-1]).to(DEVICE)
    optimizer = torch.optim.Adam(brain_supervision.parameters(), lr = 2e-5)
    for epoch in range(args.epoch):
        epoch_train(brain_supervision, optimizer, train_dl, log_path, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='brain-in-the-loop supervision')
    parser.add_argument("--proxy", type=str, default="CLIP", help="model type")
    parser.add_argument("--model_size", type=str, default='ViT-B-16', help="model size")
    parser.add_argument("--sub", type=str, default='sub-1', help="DIR subject id")
    parser.add_argument("--rois", type=str, default='HVC', help="brain area")
    parser.add_argument("--batch_size", type=int, default=256, help="")
    parser.add_argument("--epoch", type=int, default=20, help="")
    args = parser.parse_args()

    main(args)