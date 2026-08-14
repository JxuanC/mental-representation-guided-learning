import clip
import torch
from modules.LSP.encoder import lsp
import matplotlib.pyplot as plt
from skimage.transform import resize
from torchvision import transforms
from data.DIR import DIR_sub_with_images, DIR_sub_with_images_nopad
from data.data_augment import get_fmri_image_dataset
import data.data_config as cog
from modules.LSP.optimal_Transport import log_optimal_transport
from modules.matching_evaluation import cal_topk_acc
import argparse, os
import numpy as np
import glob
import math
from himalaya.backend import set_backend
from himalaya.ridge import RidgeCV
from himalaya.scoring import correlation_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

def ridgeregression(X, Y, X_te):
    ridge = RidgeCV()
    preprocess_pipeline = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
    )
    pipeline = make_pipeline(
        preprocess_pipeline,
        ridge,
    )
    pipeline.fit(X, Y)
    return pipeline.predict(X_te)

def ridge_test(selected_rois, selected_sub):
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 100
    vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
    trainRois, trainStis, trainStiIDs, testRois, testStis, testStiIDs = DIR_sub_with_images_nopad(preprocess, sub = selected_sub, rois = selected_rois)
    X = trainRois
    X_te = testRois
    Y = []
    with torch.no_grad():
        for idx in range(math.ceil(trainStis.shape[0]/BATCH_SIZE)):
            Y.append(vit_clip.encode_image(trainStis[idx*BATCH_SIZE:(idx+1)*BATCH_SIZE].to(DEVICE)).float())
        Y_te = vit_clip.encode_image(testStis.to(DEVICE)).float()
        Y = torch.cat(Y, 0)
    Y_pre = ridgeregression(X, Y.cpu().numpy(), X_te)

    Y_pre = torch.tensor(Y_pre).to(DEVICE) / torch.tensor(Y_pre).to(DEVICE).norm(dim=-1, keepdim=True)
    Y_te = Y_te / Y_te.norm(dim=-1, keepdim=True)
    similarity = torch.matmul(torch.tensor(Y_pre).to(DEVICE).float(), Y_te.float().T)
    acc_table = torch.max(similarity.softmax(dim=-1), -1)[1] == torch.tensor(range(Y_pre.shape[0])).to(DEVICE)
    top1_acc = torch.sum(acc_table) / 50
    print(f"Test ==> Top1 acc: {(top1_acc * 100):.2f}%")

    return Y_pre, Y_te
    

def brain_in_the_loop_test(selected_rois, selected_dataset, selected_sub, model_path):
    DEVICE = "cuda:3" if torch.cuda.is_available() else "cpu"
    vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
    trainRois, trainStis, trainStiIDs, testRois, testStis, testStiIDs = DIR_sub_with_images(preprocess, sub = selected_sub, rois = selected_rois)
    train_dl, test_dl, fmri_dim = get_fmri_image_dataset(selected_dataset, selected_sub, selected_rois, 50, 0, preprocess)
    lsp_model = lsp(fmri_dim, len(selected_rois), 0, 512, 12, 8, None, None).to(DEVICE)
    lsp_model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
    lsp_model.eval()
    with torch.no_grad():
        for idx, (images, fmris, img_paths) in enumerate(test_dl):
            images = images.to(DEVICE)
            fmris = fmris.float().to(DEVICE)
            clip_embedding = vit_clip.encode_image(testStis.to(DEVICE)).float()
            fmri_embedding, img_embedding = lsp_model(fmris, clip_embedding)
            fmri_embedding = fmri_embedding / fmri_embedding.norm(dim=-1, keepdim=True)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            clip_embedding = clip_embedding / clip_embedding.norm(dim=-1, keepdim=True)
            similarity = torch.matmul(fmri_embedding, clip_embedding.float().T)
        acc_table = torch.max(similarity.softmax(dim=-1), -1)[1] == torch.tensor(range(fmris.shape[0])).to(DEVICE)
        top1_acc = torch.sum(acc_table) / 50
        print(f"Test ==> Top1 acc: {(top1_acc * 100):.2f}%")
        OTscores = log_optimal_transport(similarity.unsqueeze(0), 
                                        torch.nn.Parameter(torch.tensor(1.0)).to(similarity), iters = 100)
        #print(f"OT Top 1 acc: {cal_topk_acc(OTscores[0,:,:], torch.eye(OTscores.shape[-1], dtype=torch.bool))}")
    return fmri_embedding, img_embedding, clip_embedding


def main(args):
    selected_rois,selected_dataset, selected_sub = cog.HVC, 'DIR', args.subid
    ridge_test(selected_rois, selected_sub)
    brain_in_the_loop_test(selected_rois, selected_dataset, selected_sub, args.model_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='brain2image')
    parser.add_argument("--model_path", type=str, help="model_path")
    parser.add_argument("--subid", type=str, default='sub-3', help="DIR subid")
    args = parser.parse_args()

    main(args)