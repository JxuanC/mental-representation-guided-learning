import os
import bdpy
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
from data.DIR import DIR_sub_without_images
import torch.utils.data
import data.data_config as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio


class Image_Dataset(Dataset):
     def __init__(self, imageIDs, clip_feature = None, transform = None):
         self.imageIDs = imageIDs
         self.transform = transform
         self.clip_feature = clip_feature

     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        #category_id = file_name.split('_')[0]
        category_id = self.imageIDs[idx].split('/')[-2]
        image_id = file_name.split('.')[0]
        if(self.clip_feature is None):
            image = Image.open(self.imageIDs[idx]).convert("RGB")
            image = self.transform(image) if self.transform else image
            return image, file_name, category_id
        else:
            return self.clip_feature[image_id][:], file_name, category_id


class Image_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, fMRI, mixup = False, train = True, transform = None):
         self.imageIDs = imageIDs
         self.fMRI = fMRI
         self.mixup = mixup
         self.train = train
         self.transform = transform
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image = Image.open(self.imageIDs[idx])

        if(self.train):
            fmri_num = self.fMRI[category_id].shape[0]
            selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
            if(self.mixup and selected_no.shape[0] != 1):
                coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0])).softmax(0)
                mixup_fMRI = torch.sum(torch.tensor(self.fMRI[category_id][selected_no]) * coefficient[:, None, None], 0)
                fMRI = mixup_fMRI.numpy()
            else:
                fMRI = self.fMRI[category_id][selected_no[0]].squeeze()
        else:
            selected = random.randint(0, self.fMRI[category_id].shape[0] - 1)
            fMRI = self.fMRI[category_id][selected]

        image = self.transform(image) if self.transform else image
        return image, fMRI, category_id
     
class fMRI_Image_Dataset(Dataset):
     def __init__(self, Aug_data_path, fMRI, fMRI_StiIDs, augment_num = 9, transform = None):
         self.Aug_data_path = Aug_data_path
         self.fMRI = fMRI
         self.fMRI_StiIDs = fMRI_StiIDs
         self.transform = transform
         self.augment_num = augment_num
     def __len__(self):
         return len(self.fMRI)

     def __getitem__(self, idx):
        fMRI = self.fMRI[idx]
        fMRI_StiID = self.fMRI_StiIDs[idx]
        sti_category_id = fMRI_StiID.split('_')[0]
        sti_image_path = f"{config.kamitani_Aug}/{sti_category_id}/{fMRI_StiID}"
        aug_image_path = glob.glob(f"{self.Aug_data_path}/{sti_category_id}/*.JPEG")
        aug_image_path.remove(sti_image_path)
        random_index = torch.randperm(len(aug_image_path))[:self.augment_num]
        selected_aug_images = np.array(aug_image_path)[random_index]

        sti_image = [self.transform(Image.open((sti_image_path)))[None,:,:,:]]
        aug_images = [self.transform(Image.open(path))[None,:,:,:] for path in selected_aug_images]
        images = torch.cat(sti_image + aug_images)
        return images, fMRI, [fMRI_StiID] + [path.split('augment/')[-1].split('/')[-1] for path in selected_aug_images]

class CLIP_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, CLIP_features, fMRI, mixup = False, train = True, random_matching = False):
         self.imageIDs = imageIDs
         self.CLIPv_features = CLIP_features[0]
         self.CLIPt_features = CLIP_features[1]
         self.fMRI = fMRI
         self.mixup = mixup
         self.train = train
         self.random_matching = random_matching
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        image_category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image_clip = self.CLIPv_features[idx]
        text_clip = self.CLIPt_features[image_category_id][:]
        #image = Image.open(self.imageIDs[idx])
        #vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)

        fmri_category_id = np.random.choice(list(self.fMRI.keys())) if self.random_matching else image_category_id

        if(self.train):
            fmri_num = self.fMRI[fmri_category_id].shape[0]
            selected_no = np.random.permutation(range(fmri_num))[:random.randint(1, fmri_num - 1)]
            if(self.mixup and selected_no.shape[0] != 1):
                coefficient = torch.tensor(np.random.uniform(-1, 1, size = selected_no.shape[0])).softmax(0)
                mixup_fMRI = torch.sum(torch.tensor(self.fMRI[fmri_category_id][selected_no]) * coefficient[:, None, None], 0)
                fMRI = mixup_fMRI.numpy()
            else:
                fMRI = self.fMRI[fmri_category_id][selected_no[0]].squeeze()
        else:
            selected = random.randint(0, self.fMRI[fmri_category_id].shape[0] - 1)
            fMRI = self.fMRI[fmri_category_id][selected]

        return image_clip, text_clip, fMRI, image_category_id, fmri_category_id

class CLIP_fMRI_Dataset_New(Dataset):
     def __init__(self, imageIDs, CLIP_features, fMRI, train = True, random_matching = False):
         self.imageIDs = imageIDs
         self.CLIP_features = CLIP_features
         self.fMRI = fMRI
         self.train = train
         self.random_matching = random_matching
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        image_category_id = file_name.split('_')[0]
        image_id = file_name.split('_')[1].split('.')[0]
        image_clip = self.CLIP_features[idx]
        #image = Image.open(self.imageIDs[idx])
        #vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
        fmri_category_id = np.random.choice(list(self.fMRI.keys())) if self.random_matching else image_category_id
        fMRI = self.fMRI[fmri_category_id]
        return image_clip, fMRI, image_category_id, fmri_category_id


class Visual_fMRI_Dataset(Dataset):
     def __init__(self, imageIDs, visual_features, fMRI, train = True, random_matching = False):
         self.imageIDs = imageIDs
         self.visual_features = visual_features
         self.fMRI = fMRI
         self.train = train
         self.random_matching = random_matching
         
     def __len__(self):
         return len(self.imageIDs)

     def __getitem__(self, idx):
        file_name = self.imageIDs[idx].split('/')[-1]
        image_category_id = file_name.split('_')[0]
        image_id = file_name.split('.')[0]
        image_embedding = self.visual_features[image_id][:]
        #image = Image.open(self.imageIDs[idx])
        #vit_clip, preprocess = clip.load("ViT-B/16", device = DEVICE)
        fmri_category_id = np.random.choice(list(self.fMRI.keys())) if self.random_matching else image_category_id
        fMRI = self.fMRI[fmri_category_id]
        return image_embedding, fMRI, image_category_id, fmri_category_id
          
def get_fmri_image_dataset(dataset, sub, rois, batch_size, aug_img_num = 9, img_transform = None):
    if(dataset == 'DIR'):
         _, train_rois, trainStiIDs, _, test_rois, testStiIDs = DIR_sub_without_images(sub, rois)
    else:
        raise NotImplementedError
    train_dataset = fMRI_Image_Dataset(config.kamitani_Aug, train_rois, trainStiIDs, aug_img_num, img_transform)
    test_dataset = fMRI_Image_Dataset(config.kamitani_Aug, test_rois, testStiIDs, aug_img_num, img_transform)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = False)
    return train_dataloader, test_dataloader, train_rois.shape[-1]

def get_image_fmri_dataset(dataset, sub, rois, batch_size, mixup = True, candidate = True, img_transform = None, clip_features = None):
    if(dataset == 'DIR'):
        train_cat_rois, _, trainStiIDs, test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)
    else:
        raise NotImplementedError
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Train_category])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in trainStiIDs])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in testStiIDs])
    if(clip_features is None):
        train_dataset = Image_fMRI_Dataset(train_images, train_cat_rois, mixup, True, img_transform)
        test_dataset = Image_fMRI_Dataset(test_images, test_cat_rois, mixup, False, img_transform)
    else:
        train_dataset = CLIP_fMRI_Dataset(np.sort(train_images), [clip_features[0], clip_features[-1]], train_cat_rois, mixup, True, False)
        test_dataset = CLIP_fMRI_Dataset(np.sort(test_images), [clip_features[1], clip_features[-1]], test_cat_rois, mixup, False, False)
        
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader, test_dataloader, fmri_dim

def get_clip_fmri_dataset(fmri_dataset_name, sub, rois, clip_dataset, batch_size, 
                          mixup = True, candidate = True, random_matching = False):
    if(fmri_dataset_name == 'DIR'):
        train_cat_rois, _, trainStiIDs, test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)
    else:
        raise NotImplementedError
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        #trainimageIDs = pd.read_csv(config.kamitani_sti_trainID, header = None)
        #testimageIDs = pd.read_csv(config.kamitani_sti_testID, header = None)
        #Train_category = set([id.split('_')[0] for id in list(trainimageIDs[1])])
        #Test_category = set([id.split('_')[0] for id in list(testimageIDs[1])])
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Train_category])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in trainStiIDs])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in testStiIDs])
        
    train_images = np.hstack((np.sort(train_images), np.sort(test_images))) if random_matching else np.sort(train_images)
    test_images = np.sort(test_images)
    #train_dataset = CLIP_fMRI_Dataset(np.sort(train_images), clip_dataset[0], train_cat_rois, mixup, True, random_matching)
    #test_dataset = CLIP_fMRI_Dataset(np.sort(test_images), clip_dataset[1], test_cat_rois, mixup, False, random_matching)
    train_dataset = CLIP_fMRI_Dataset_New(train_images, clip_dataset[0], train_cat_rois, True, random_matching)
    test_dataset = CLIP_fMRI_Dataset_New(test_images, clip_dataset[1], test_cat_rois, False, random_matching)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader, test_dataloader, fmri_dim


def get_visual_fmri_dataset(fmri_dataset_name, sub, rois, clip_dataset, batch_size, 
                          mixup = True, candidate = True, random_matching = False):
    if(fmri_dataset_name == 'DIR'):
        train_cat_rois, _, trainStiIDs, test_cat_rois, _, testStiIDs = DIR_sub_without_images(sub, rois)
    else:
        raise NotImplementedError
    fmri_dim = train_cat_rois[trainStiIDs[0].split('_')[0]].shape[-1]
    Train_category = set([id.split('_')[0] for id in trainStiIDs])
    Test_category = set([id.split('_')[0] for id in testStiIDs])
    if(candidate):
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Train_category])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/{category}/*.JPEG")) for category in Test_category])
    else:
        train_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in trainStiIDs])
        test_images = np.concatenate([np.array(glob.glob(f"{config.kamitani_Aug}/*/{image}")) for image in testStiIDs])
        
    train_images = np.hstack((train_images, test_images)) if random_matching else train_images

    train_dataset = Visual_fMRI_Dataset(train_images, clip_dataset[0], train_cat_rois, True, random_matching)
    test_dataset = Visual_fMRI_Dataset(test_images, clip_dataset[1], test_cat_rois, False, random_matching)
    train_dataloader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True)
    test_dataloader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)
    return train_dataloader, test_dataloader, fmri_dim