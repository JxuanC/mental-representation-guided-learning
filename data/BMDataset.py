import os
import clip
import bdpy
import json
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image
import torch.utils.data
import data.data_config as config
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
import torchvision.io.image as imageio

class BMDataset(Dataset):
     def __init__(self, fmri, text, imgids,  clip_features, label_process = None):
         self.imgids = imgids
         self.fmris = fmri
         self.texts = text
         self.clip_features = clip_features
         self.label_process = label_process

     def __len__(self):
         return len(self.imgids)

     def __getitem__(self, idx):
        img_id = self.imgids[idx].split('/')[-1]
        cat_id = img_id.split('_')[0]
        cat_text = self.text[img_id]
        visual_feature = self.clip_features[img_id]
        fmri = self.fmris[cat_id]
        label = self.label_process(fmri)
        
        return {'visual_feature': visual_feature, 
                'cat_text': cat_text, 
                'label': label, 
                'fmri': fmri,
                'img_id': img_id}