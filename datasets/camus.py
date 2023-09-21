import os
import random
import numpy as np

import torch
import torch.nn as nn
import torchvision.transforms as transforms

import nibabel as nib
import cv2

import SimpleITK as sitk
from PIL import Image
from collections import defaultdict
from torch.utils.data import Dataset

from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    RandSpatialCropd,
    CenterSpatialCropd,
    EnsureChannelFirstd,
    RandFlipd,
    Resized,
    ScaleIntensityRangePercentilesd,
    NormalizeIntensityd,
    Identity,
    EnsureTyped
)

from glob import glob
import skimage.io as io

NUM_PREFETCH = 10
RANDOM_SEED = 123


class DataLoaderCamus(Dataset):
    def __init__(self, dataset_path, input_name, target_name, condition_name, stage, single_frame=True,
                 img_res=(272, 272), img_crop=(256, 256), seg_parts=True, train_ratio=1.0, valid_ratio=0.2):
        self.dataset_path = dataset_path
        self.input_name = input_name
        self.target_name = target_name
        self.condition_name = condition_name
        self.spatial_size = img_res[0]
        self.crop_size = img_crop[0]
        self.single_frame = single_frame
        self.seg_parts = seg_parts
        self.is_train = True if stage == 'train' else False

        self.transform = self.get_transform(self.is_train)
        
        patients = []
        for id, dir in enumerate(sorted(glob(os.path.join(self.dataset_path, 'training', '*')))):
            if not os.listdir(dir):
                continue
            else:
                patients.append(dir)

        random.Random(RANDOM_SEED).shuffle(patients)
        num = len(patients)
        num_train = int(num * train_ratio)
        num_valid = int(num_train * valid_ratio)

        self.train_patients = patients[num_valid:num_train]
        self.valid_patients = patients[:num_valid//2]
        self.test_patients = patients[num_valid//2:num_valid]
        #self.test_patients = patients[num_train:]
        '''
        if train_ratio == 1.0:
            self.test_patients = glob(os.path.join(self.dataset_path, 'testing', '*'))
        '''
        if stage == 'train':
            self.data_list = self.train_patients
            print('#train:', len(self.data_list))
        elif stage == 'valid':
            self.data_list = self.valid_patients
            print('#valid:', len(self.data_list))
        elif stage == 'test':
            self.data_list = self.test_patients
            print('#test:', len(self.data_list))

        self.data_length = len(self.data_list)

    def __getitem__(self, index):
        mask_index = 0
        path = self.data_list[index]
        input_path, condition_path = self.get_path(path)
        while not os.path.exists(input_path):
            index = random.randint(0, self.data_length)
            path = self.data_list[index]
            input_path, condition_path = self.get_path(path)
        
        input_img = self.read_mhd(input_path, '_gt' in self.input_name)
        condition_img = self.read_mhd(condition_path, True)

        if self.seg_parts:
            LV = np.where(condition_img == 1, 1, 0)
            LA = np.where(condition_img == 3, 1, 0)
            condition_img = np.stack([LV, LA], axis=0)

        input_dict = self.transform({'images': input_img, 'masks': condition_img})

        return input_dict['images'] / 255.0, input_dict['masks'] / 1.0, mask_index, index 

    def __len__(self):
        return len(self.data_list)

    def read_mhd(self, img_path, is_gt):
        img = io.imread(img_path, plugin='simpleitk').squeeze()
        return img

    def get_path(self, img_path):
        _, patient_id = os.path.split(img_path)
        input_path = os.path.join(img_path, '{}_{}.mhd'.format(patient_id, self.input_name))
        condition_path = os.path.join(img_path, '{}_{}.mhd'.format(patient_id, self.condition_name))

        return input_path, condition_path

    def get_transform(self, is_train):
        all_keys = ['images', 'masks']
        
        if self.single_frame:
            spatial_size = (self.spatial_size, self.spatial_size)
            crop_size = (self.crop_size, self.crop_size)
        else:
            spatial_size = (self.spatial_size, self.spatial_size, self.clip_length)
            crop_size = (self.crop_size, self.crop_size, self.clip_length)
        
        if is_train:
            
            rf0 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=0)
            rf1 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=1)
            rf2 = RandFlipd(keys=all_keys, prob=0.5, spatial_axis=2) if not self.single_frame else None

            rf0.set_random_state(0)
            rf1.set_random_state(0)
            if rf2 is not None:
                rf2.set_random_state(0)

            transform = Compose([
                    AddChanneld(keys=['images'] if self.seg_parts else all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                    RandSpatialCropd(all_keys, crop_size, random_size=False, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        else:
            transform = Compose([
                    AddChanneld(keys=['images'] if self.seg_parts else all_keys, allow_missing_keys=True),
                    Resized(keys=all_keys, spatial_size=spatial_size, allow_missing_keys=True, mode='nearest'),
                    CenterSpatialCropd(keys=all_keys, roi_size=crop_size, allow_missing_keys=True),
                    #ScaleIntensityRangePercentilesd(keys=all_keys, lower=5, upper=95, b_min=0., b_max=1., allow_missing_keys=True) if len(all_keys)>0 else Identity(),
                    #NormalizeIntensityd(keys=all_keys, subtrahend=0.5, divisor=0.5),
                    EnsureTyped(keys=all_keys, allow_missing_keys=True),
                ])
        return transform

if __name__ == '__main__':
    data_loader = DataLoaderCamus(
            dataset_path='/home/jyangcu/Dataset/camus_dataset',
            input_name="4CH_ED",
            target_name="4CH_ED",
            condition_name="4CH_ED_gt",
            stage="train",
        )
    from monai.data import DataLoader
    train_loader = DataLoader(data_loader, batch_size=2, shuffle=False, num_workers=1)

    for targets, targets_gt, _, _ in train_loader:
        print(targets.shape)
        print(targets_gt.shape)