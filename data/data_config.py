
kamitani_Aug = f'./dataset/ImageNet/'
kamitani_Sti = f'./dataset/DIR/stimulus'

DIR_dir = f"./dataset/fMRI/DeepImageReconstruction/"

kamitani_sti_trainID = f"./dataset/imageID_training.csv"
kamitani_sti_testID = f"./dataset/imageID_test.csv"

DIR_train_subs = {
    'sub-1':'sub-01_perceptionNaturalImageTraining_VC_v2.h5',
    'sub-2':'sub-02_perceptionNaturalImageTraining_VC_v2.h5',
    'sub-3':'sub-03_perceptionNaturalImageTraining_VC_v2.h5'
}

DIR_test_subs = {
    'sub-1':'sub-01_perceptionNaturalImageTest_VC_v2.h5',
    'sub-2':'sub-02_perceptionNaturalImageTest_VC_v2.h5',
    'sub-3':'sub-03_perceptionNaturalImageTest_VC_v2.h5'
}

roi_list = {
    'VC':  'ROI_VC',
    'LVC': 'ROI_LVC',
    'HVC': 'ROI_HVC',
    'V1':  'ROI_V1',
    'V2':  'ROI_V2',
    'V3':  'ROI_V3',
    'V4':  'ROI_V4',
    'LOC': 'ROI_LOC',
    'FFA': 'ROI_FFA',
    'PPA': 'ROI_PPA',
}

FVC = ['ROI_V1', 'ROI_V2', 'ROI_V3', 'ROI_V4', 'ROI_LOC', 'ROI_FFA', 'ROI_PPA']
LVC = ['ROI_V1', 'ROI_V2', 'ROI_V3']
HVC = ['ROI_LOC', 'ROI_FFA', 'ROI_PPA']

ROIS = {'FVC': FVC,
       'LVC': LVC,
       'HVC': HVC}