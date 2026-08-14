import os
#import faiss
import clip
import torch
import h5py
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import torchvision.models as models
from torch import nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
import argparse

MODEL_SIZE = { "dinov2-small": "small", 
              "dinov2-base": "base",
              "dinov2-large": "large"}


class Image_Dataset(Dataset):
     def __init__(self, imageIDs, transform = None):
         self.imageIDs = imageIDs
         self.transform = transform
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        category_id = file_name.split('_')[0]
        #image_id = file_name.split('_')[1].split('.')[0]
        image = Image.open(self.imageIDs[idx]).convert("RGB")
        image = self.transform(images = image, return_tensors = 'pt').pixel_values.squeeze() if self.transform else image
        return image, file_name, category_id
     
@torch.no_grad()
def extract_image_embedding(images_dir, model_type, model_size, model, preprocess, save_name, DEVICE):
    dataloader = DataLoader(dataset = Image_Dataset(images_dir, preprocess), batch_size = 512, shuffle = False, pin_memory = True)
    if not os.path.exists(f'./pretrained-features/{model_type}/{MODEL_SIZE[model_size]}'):
        os.makedirs(f'./pretrained-features/{model_type}/{MODEL_SIZE[model_size]}')
    if os.path.exists(f"./pretrained-features/{model_type}/{MODEL_SIZE[model_size]}/{save_name}.hdf5"):
        print("hdf5 existing")
        return
    h5py_file = h5py.File(f"./pretrained-features/{model_type}/{MODEL_SIZE[model_size]}/{save_name}.hdf5", 'w')
    image_embedding = []
    for idx, (images, image_ids, category_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img_embedding = model(images.to(DEVICE)).last_hidden_state[:, 0, :].squeeze()
        image_embedding.append(img_embedding.cpu().numpy())
        for imgid, embedding in zip(image_ids, img_embedding):
            h5py_file.create_dataset(str(imgid).split('.')[0], (img_embedding.shape[-1]), data = embedding.cpu().numpy())


def main(args):
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    feature_extractor = AutoImageProcessor.from_pretrained(f'./pretrained_models/{args.model_type}/{args.model_size}')
    model = AutoModel.from_pretrained(f'./pretrained_models/{args.model_type}/{args.model_size}').to(DEVICE)

    teimageIDs = pd.read_csv('./dataset/imageID_test.csv', header = None)
    trimageIDs = pd.read_csv('./dataset/imageID_training.csv', header = None)

    TeCategory = set([id.split('_')[0] for id in list(teimageIDs[1])])
    TeImgs_dir = np.concatenate([np.array(glob.glob(f"{args.dataset_path}/ImageNet/{category}/*.JPEG")) for category in TeCategory])
    extract_image_embedding(TeImgs_dir, args.model_type, args.model_size, model, feature_extractor, f'kamitani_test', DEVICE=DEVICE)

    TrCategory = set([id.split('_')[0] for id in list(trimageIDs[1])])
    TrImgs_dir = np.concatenate([np.array(glob.glob(f"{args.dataset_path}/ImageNet/{category}/*.JPEG")) for category in TrCategory])
    extract_image_embedding(TrImgs_dir, args.model_type, args.model_size, model, feature_extractor, f'kamitani_train', DEVICE=DEVICE)
    
    if(args.extract_THINGS):
        THINGS_object_1854 = glob.glob(f"{args.dataset_path}/THINGS/*.jpg")
        extract_image_embedding(THINGS_object_1854, args.model_type, args.model_size, model, feature_extractor, f'THINGS_object_1854', DEVICE=DEVICE)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DINOv2')
    parser.add_argument("--model_type", type=str, default="DINOv2", help="model type")
    parser.add_argument("--model_size", type=str, default='dinov2-base', help="model size")
    parser.add_argument("--dataset_path", type=str, default='./dataset/', help="dataset path")
    parser.add_argument("--extract_THINGS", type=bool, default=False, help="Whether to extract features from THINGS")
    args = parser.parse_args()

    main(args)