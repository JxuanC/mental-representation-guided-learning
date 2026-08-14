import clip
import torch
import numpy as np
import matplotlib.pyplot as plt
from data.data_augment import Image_Dataset
from torchvision import transforms
from modules.LSP.encoder import lsp
import data.data_config as cog
from torch.utils.data import DataLoader
import pandas as pd
import glob
from tqdm import tqdm
import h5py
import numpy as np
import json
from PIL import Image
import os
import argparse

data_dir = './dataset/coco/'
features_dir = './pretrained_features/'

annotations = json.load(open('./dataset/dataset_coco.json'))['images']

def load_coco_data(key = 'clip_coco'):
    data = {f'{key}_train': [], f'{key}_val': []}
    for item in annotations:
        file_name = item['filename'].split('_')[-1]
        if item['split'] == 'train' or item['split'] == 'restval':
            data[f'{key}_train'].append({'file_name': file_name, 'cocoid': item['cocoid']})
        elif item['split'] == 'val':
            data[f'{key}_val'].append({'file_name': file_name, 'cocoid': item['cocoid']})
    return data

def encode_clip_split(data, split, DEVICE):
    vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
    df = pd.DataFrame(data[split])
    bs = 512
    if os.path.exists(features_dir + '{}.hdf5'.format(split)):
        print("hdf5 existing")
        return
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['cocoid'][idx:idx + bs]
        file_names = df['file_name'][idx:idx + bs]
        images = [preprocess(Image.open(data_dir + file_name).convert("RGB"))[None,] for file_name in file_names]
        with torch.no_grad(): 
            images = torch.cat(images)
            encodings = vit_clip.encode_image(images.to(DEVICE)).cpu().numpy() 
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(str(cocoid), (encodings.shape[-1]), data=encoding)
    return h5py_file

def encode_lsp_split(model, data, clip_features, split, DEVICE):
    df = pd.DataFrame(data[split])
    bs = 10240
    if os.path.exists(features_dir + '{}.hdf5'.format(split)):
        print("hdf5 existing")
        return
    h5py_file = h5py.File(features_dir + '{}.hdf5'.format(split), 'w')
    for idx in tqdm(range(0, len(df), bs)):
        cocoids = df['cocoid'][idx:idx + bs]
        clip_embedding = [torch.tensor(clip_features[str(cocoid)][:][None]) for cocoid in cocoids]
        with torch.no_grad(): 
            encodings = model.encode_image(torch.cat(clip_embedding).to(DEVICE).float()).cpu().numpy() 
        for cocoid, encoding in zip(cocoids, encodings):
            h5py_file.create_dataset(str(cocoid), (encodings.shape[-1]), data=encoding)

def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

    fmri_dim = 3745
    selected_dataset, selected_sub, selected_rois = "DIR", "sub-3", cog.HVC
    # vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
    model_path = args.pretrained_brain_model
    brain_model = lsp(fmri_dim, len(selected_rois), 0, 512, 12, 8, None, None).to(DEVICE)
    brain_model.load_state_dict(torch.load(model_path, map_location = 'cpu'))
    brain_model.eval()

    clip_coco_train = encode_clip_split(load_coco_data('clip_coco'), f'clip_coco_train', DEVICE)
    clip_coco_val = encode_clip_split(load_coco_data('clip_coco'), f'clip_coco_val', DEVICE)
    encode_lsp_split(brain_model, load_coco_data('lsp_coco'), clip_coco_train, f'lsp_coco_train', DEVICE)
    encode_lsp_split(brain_model, load_coco_data('lsp_coco'), clip_coco_val, f'lsp_coco_val', DEVICE)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract brain-in-the-loop features')
    parser.add_argument("--pretrained_brain_model", type=str, help="pretrained_brain_model")
    args = parser.parse_args()

    main(args)
