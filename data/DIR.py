import os
import bdpy
import torch
import numpy as np
import pandas as pd
from PIL import Image
import data.data_config as config

def DIR_sub_with_images(img_transfomers, sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois, test_avg_rois = [], [], []

    DIR_sub_train = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_train_subs[sub]))
    DIR_sub_test = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_test_subs[sub]))

    train_image_index = DIR_sub_train.select('image_index').squeeze().astype(int) - 1
    test_image_index = DIR_sub_test.select('image_index').squeeze().astype(int) - 1

    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    trainStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/train/{file}"))[None,:,:,:] for file in trainStiIDs])
    testStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/test/{file}"))[None,:,:,:] for file in testStiIDs])

    MAX_DIM = 0
    for roi in rois:
        train_roi_fMRI = DIR_sub_train.select(roi)
        test_roi_fMRI = DIR_sub_test.select(roi)

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI)
        test_avg_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()
    test_avg_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_avg_rois]), 1).squeeze()

    return train_rois, trainStis, trainStiIDs, test_avg_rois, testStis, testStiIDs#, test_rois, test_image_index

def DIR_sub_with_images_nopad(img_transfomers, sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois, test_avg_rois = [], [], []

    DIR_sub_train = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_train_subs[sub]))
    DIR_sub_test = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_test_subs[sub]))

    train_image_index = DIR_sub_train.select('image_index').squeeze().astype(int) - 1
    test_image_index = DIR_sub_test.select('image_index').squeeze().astype(int) - 1

    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    trainStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/train/{file}"))[None,:,:,:] for file in trainStiIDs])
    testStis = torch.cat([img_transfomers(Image.open(f"{config.kamitani_Sti}/test/{file}"))[None,:,:,:] for file in testStiIDs])

    MAX_DIM = 0
    for roi in rois:
        train_roi_fMRI = DIR_sub_train.select(roi)
        test_roi_fMRI = DIR_sub_test.select(roi)

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI)
        test_avg_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(train_rois, 1).squeeze()
    test_rois = np.concatenate(test_rois, 1).squeeze()
    test_avg_rois = np.concatenate(test_avg_rois, 1).squeeze()

    return train_rois, trainStis, trainStiIDs, test_avg_rois, testStis, testStiIDs#, test_rois, test_image_index

def DIR_sub_without_images(sub = 'sub-3', rois = ['ROI_VC']):
    train_rois, test_rois = [], []
    DIR_sub_train = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_train_subs[sub]))
    DIR_sub_test = bdpy.BData(os.path.join(config.DIR_dir, config.DIR_test_subs[sub]))

    train_image_index = DIR_sub_train.select('image_index').squeeze().astype(int) - 1
    test_image_index = DIR_sub_test.select('image_index').squeeze().astype(int) - 1

    trainStiIDs = np.array(pd.read_csv(config.kamitani_sti_trainID, header = None)[1])[train_image_index]
    testStiIDs = np.array(pd.read_csv(config.kamitani_sti_testID, header = None)[1])
    
    MAX_DIM = 0
    for roi in rois:
        train_roi_fMRI = DIR_sub_train.select(roi)
        test_roi_fMRI = DIR_sub_test.select(roi)

        test_roi_fMRI_avg = np.zeros([50, test_roi_fMRI.shape[1]])
        for i in range(50):
            test_roi_fMRI_avg[i] = np.mean(test_roi_fMRI[test_image_index == i], axis = 0)

        train_rois.append(train_roi_fMRI)
        test_rois.append(test_roi_fMRI_avg)
        MAX_DIM = train_roi_fMRI.shape[-1] if train_roi_fMRI.shape[-1] > MAX_DIM else MAX_DIM

    train_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in train_rois]), 1).squeeze()
    test_rois = np.concatenate(([np.pad(fmri, ((0, 0), (0, MAX_DIM - fmri.shape[-1])))[:,None,:] for fmri in test_rois]), 1).squeeze()

    train_cat_rois = {}
    trainCatIDs = [id.split('_')[0] for id in trainStiIDs]
    trainCatSet = set(trainCatIDs)
    for cat in trainCatSet:
        train_cat_rois[cat] = train_rois[np.array(trainCatIDs) == cat]

    test_cat_rois = {}
    testCatIDs = [id.split('_')[0] for id in testStiIDs]
    for cat in testCatIDs:
        test_cat_rois[cat] = test_rois[np.array(testCatIDs) == cat]

    return train_cat_rois, train_rois, trainStiIDs, test_cat_rois, test_rois, testStiIDs