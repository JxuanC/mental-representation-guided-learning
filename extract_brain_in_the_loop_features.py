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
import argparse

def main(args):
    fmri_dim = {'sub-3': 3745, 'sub-2': 2627, 'sub-1': 3072}
    selected_dataset, selected_sub, selected_rois = 'DIR', args.subid, cog.HVC
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
    image_feature = h5py.File(args.pretrained_features, 'r')
    model_path = args.pretrained_brain_model
    brain_lsp = lsp(fmri_dim[selected_sub], len(selected_rois), 0, 
                        image_feature['n01621127_11310'].shape[0], 12, 8, None, None, visual_dim = image_feature['n01621127_11310'].shape[0]).to(DEVICE)
    
    brain_lsp.load_state_dict(torch.load(model_path, map_location = 'cpu'))
    brain_lsp.eval()
    imageIDs = pd.read_csv('./dataset/imageID_test.csv', header = None)
    Category = set([id.split('_')[0] for id in list(imageIDs[1])])
    Imgs_dir = np.concatenate([np.array(glob.glob(f"{cog.kamitani_Aug}/{category}/*.JPEG"))[:args.image_per_class] for category in Category])
    dataloader = DataLoader(dataset = Image_Dataset(Imgs_dir, image_feature), batch_size = 10240, shuffle = False, pin_memory = True)

    without_brain_features = []
    with_brain_features = []
    category_label = []
    with torch.no_grad():
        for idx, (img_embedding, image_ids, category_ids) in tqdm(enumerate(dataloader), total = len(dataloader)):
            lsp_embedding = brain_lsp.encode_image(img_embedding.to(DEVICE).float())
            lsp_embedding = lsp_embedding / lsp_embedding.norm(dim=-1, keepdim=True)
            img_embedding = img_embedding / img_embedding.norm(dim=-1, keepdim=True)
            without_brain_features.append(img_embedding)
            with_brain_features.append(lsp_embedding)
            category_label.append(category_ids)
    with_brain_features = torch.cat(without_brain_features)
    without_brain_features = torch.cat(with_brain_features)
    category_label = np.hstack(category_label)

    np.savez(f'./pretrained_features/brain_in_the_loop_supervision_features.npz', with_brain_features = with_brain_features,
                                                                                  without_brain_features = without_brain_features,                                    
                                                                                  feature_labels = category_label)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract brain-in-the-loop features')
    parser.add_argument("--pretrained_brain_model", type=str, help="pretrained_brain_model")
    parser.add_argument("--pretrained_features", type=str, default='./pretrained_features/CLIP/ViT-B-16/kamitani_test.hdf5', help="proxy features")
    parser.add_argument("--subid", type=str, default='sub-3', help="DIR subjects")
    parser.add_argument("--image_per_class", type=int, default=275, help="image number per class")
    args = parser.parse_args()

    main(args)