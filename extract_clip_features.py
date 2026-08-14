import os
import faiss
import clip
import torch
import h5py
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import argparse

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
# CLIP_MODEL = { "ViT-B/16": "ViT-B-16", 
#               "ViT-L/14": "ViT-L-14"}

MODEL_SIZE = { "ViT-B/16": "base", 
              "ViT-L/14": "large"}

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
        image = self.transform(image) if self.transform else image
        return image, file_name, category_id
     
@torch.no_grad()
def extract_clipv_embedding(images_dir, clip_model, save_name, save_path = './pretrained_features'):
    vit_clip, preprocess = clip.load(clip_model, device = DEVICE)
    dataloader = DataLoader(dataset = Image_Dataset(images_dir, preprocess), batch_size = 1024, shuffle = False, pin_memory = True)
    if not os.path.exists(f"{save_path}/CLIP/{MODEL_SIZE[clip_model]}/"):
        os.makedirs(f"{save_path}/CLIP/{MODEL_SIZE[clip_model]}/")

    if os.path.exists(f"{save_path}/CLIP/{MODEL_SIZE[clip_model]}/{save_name}.hdf5"):
        print("hdf5 existing")
        return

    h5py_file = h5py.File(f"{save_path}/CLIP/{MODEL_SIZE[clip_model]}/{save_name}.hdf5", 'w')
    CLIP_image_embedding = []
    
    for idx, (images, image_ids, category_ids) in tqdm(enumerate(dataloader), total=len(dataloader)):
        img_embedding = vit_clip.encode_image(images.to(DEVICE))
        CLIP_image_embedding.append(img_embedding.cpu().numpy())
        for imgid, embedding in zip(image_ids, img_embedding):
            h5py_file.create_dataset(str(imgid).split('.')[0], (img_embedding.shape[-1]), data = embedding.cpu().numpy()) 
    print("successful")


def main(args):
    imageTraIDs = pd.read_csv('./dataset/imageID_training.csv', header = None)
    TraCategory = set([id.split('_')[0] for id in list(imageTraIDs[1])])
    TraImgs_dir = np.concatenate([np.array(glob.glob(f"{args.dataset_path}/ImageNet/{category}/*.JPEG")) for category in TraCategory])
    #imgs_dir = np.sort(np.array(glob.glob(f"{config.kamitani_Aug}/*/*.JPEG")))
    extract_clipv_embedding(TraImgs_dir, args.model_size, 'kamitani_train')

    imageTeIDs = pd.read_csv('./dataset/imageID_test.csv', header = None)
    TeCategory = set([id.split('_')[0] for id in list(imageTeIDs[1])])
    TestImgs_dir = np.concatenate([np.array(glob.glob(f"{args.dataset_path}/ImageNet/{category}/*.JPEG")) for category in TeCategory])
    extract_clipv_embedding(TestImgs_dir, args.model_size, 'kamitani_test')

    if(args.extract_THINGS):
        THINGS_object_1854 = glob.glob(f"{args.dataset_path}/THINGS/*.jpg")
        extract_clipv_embedding(THINGS_object_1854, args.model_size, 'THINGS_object_1854', build_index = False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CLIP')
    parser.add_argument("--model_type", type=str, default="CLIP", help="model type")
    parser.add_argument("--model_size", type=str, default='ViT-B/16', help="model size: ViT-B/16, ViT-L/14")
    parser.add_argument("--dataset_path", type=str, default='./dataset/', help="dataset path")
    parser.add_argument("--extract_THINGS", type=bool, default=False, help="Whether to extract features from THINGS")
    args = parser.parse_args()

    main(args)