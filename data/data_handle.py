import os
import torch
import bdpy
import numpy as np
import pandas as pd
from PIL import Image
import data.data_config as data_config
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

def get_GOD_sub(sub = 'sub-3', rois = ['ROI_VC'], return_dict = False):
    train_rois, test_rois, test_avg_rois = [], [], []
    GOD_sub = bdpy.BData(os.path.join(data_config.GOD_dir, data_config.GOD_subs[sub]))

    DataType = GOD_sub.select('DataType')
    train_index = (DataType == 1).squeeze()
    test_index = (DataType == 2).squeeze()

    image_index = GOD_sub.select('image_index')
    train_index = image_index[train_index, :].squeeze().astype(int) - 1
    test_index = image_index[test_index, :].squeeze().astype(int) - 1
    trainStiIDs = np.array(pd.read_csv(data_config.kamitani_sti_trainID, header = None)[1])[train_index]
    testavgStiIDs = np.array(pd.read_csv(data_config.kamitani_sti_testID, header = None)[1])
    testStiIDs = testavgStiIDs[test_index]

    text_labels = pd.read_csv(data_config.kamitani_sti_text, header = None)
    classIDs = np.array(text_labels[0])
    classTexts = np.array(text_labels[1])

    id_text = dict(zip(np.array(classIDs), np.array(classTexts)))
    train_texts = np.array([id_text[id.split('_')[0]] for id in trainStiIDs])
    test_avg_texts = np.array([id_text[id.split('_')[0]] for id in testavgStiIDs])
    test_texts = np.array([id_text[id.split('_')[0]] for id in testStiIDs])

    MAX_DIM = 0
    for roi in rois:
        roi_fMRI = GOD_sub.select(roi)
        train_roi_fMRI = roi_fMRI[train_index, :]
        test_roi_fMRI = roi_fMRI[test_index, :]

        test_avg_roi_fMRI = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_avg_roi_fMRI[i] = np.mean(test_roi_fMRI[test_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI)
        test_avg_rois.append(test_avg_roi_fMRI)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()
    test_avg_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_avg_rois]), 1).squeeze()

    if(return_dict):
        train_cat_rois, train_cat_texts, train_cat_imgids = {}, {}, {}
        trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
        trainCatSet = set(trainCatIDs)
        for cat in trainCatSet:
            train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]
            train_cat_texts[cat] = train_texts[np.array(trainCatIDs) == cat]
            train_cat_imgids[cat] = trainStiIDs[np.array(trainCatIDs) == cat]

        test_cat_rois, test_cat_texts, test_cat_imgids = {}, {}, {}
        test_avg_cat_rois, test_avg_cat_texts, test_avg_cat_imgids = {}, {}, {}
        testCatIDs = [id.split('_')[0] for id in testStiIDs]
        testavgCatIDs = [id.split('_')[0] for id in testavgStiIDs]
        for cat in testavgCatIDs:
            test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]
            test_cat_texts[cat] = test_texts[np.array(testCatIDs) == cat]
            test_cat_imgids[cat] = testStiIDs[np.array(testCatIDs) == cat]
            test_avg_cat_rois[cat] = test_avg_rois[np.array(testavgCatIDs) == cat]
            test_avg_cat_texts[cat] = test_avg_texts[np.array(testavgCatIDs) == cat]
            test_avg_cat_imgids[cat] = testavgStiIDs[np.array(testavgCatIDs) == cat]
            
        return {"train_fmri": train_rois, "train_texts": train_texts, "train_imgids": trainStiIDs, 
                "test_fmri": test_rois, "test_texts": test_texts, 'test_imgids': testStiIDs,
                "test_avg_fmri": test_avg_rois, "test_avg_texts": test_avg_texts, "test_avg_imgids": testavgStiIDs,
                "train_cat_fmri": train_cat_rois, "train_cat_texts": train_cat_texts, "train_cat_imgids": train_cat_imgids, 
                "test_cat_fmri": test_cat_rois, "test_cat_texts": test_cat_texts, 'test_cat_imgids': test_cat_imgids,
                "test_cat_avg_fmri": test_avg_cat_rois, "test_cat_avg_texts": test_avg_cat_texts, "test_cat_avg_imgids": test_avg_cat_imgids}
        
    else:
        return {"train_fmri": train_rois, "train_texts": train_texts, "train_imgids": trainStiIDs, 
                "test_fmri": test_rois, "test_texts": test_texts, 'test_imgids': testStiIDs,
                "test_avg_fmri": test_avg_rois, "test_avg_texts": test_avg_texts, "test_avg_imgids": testavgStiIDs}


def get_GOD_imagery_sub(sub = 'sub-3', rois = ['ROI_VC'], return_dict = False):
    train_rois, test_rois, test_avg_rois = [], [], []
    GOD_sub = bdpy.BData(os.path.join(data_config.GOD_dir, data_config.GOD_subs[sub]))
    GOD_imagery_sub = bdpy.BData(os.path.join(data_config.GOD_dir, data_config.GOD_imagery_subs[sub]))

def map_strings_to_labels(string_list):
    unique_strings = list(set(string_list))
    label_map = {string: label for label, string in enumerate(unique_strings)}
    labels = [label_map[string] for string in string_list]
    return labels, label_map

def map_imgids_to_imgs(imgdir, imgids, image_size = 128):
    loader = transforms.Compose([transforms.Resize(image_size),transforms.CenterCrop(image_size)])
    imgs = [loader(Image.open(os.path.join(imgdir, imgid.split('_')[0], imgid))) for imgid in imgids]
    return imgs