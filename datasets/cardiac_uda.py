#jyangcu@connect.ust.hk
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
from pycox.preprocessing.label_transforms import LabTransDiscreteTime

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

np.set_printoptions(threshold=np.inf)
random.seed(7777)
np.random.seed(7777)

class Seg_Cardiac_UDA_Dataset(Dataset):
    def __init__(self, infos, root, is_train, repeat=1, data_list=None, set_select=['Site_G'], view_num=['2'], 
                 spatial_size=328, crop_size=256, transform=None, single_frame=True, total_length=40, clip_length=8, seg_parts=True, source_domain=True, fill_mask=False):
        self.rort = root
        self.is_train = is_train
        self.set_select = set_select
        self.view_num = view_num
        self.spatial_size = spatial_size
        self.crop_size = crop_size
        self.single_frame = single_frame
        self.total_length = total_length
        self.clip_length = clip_length
        self.seg_parts = seg_parts
        self.source_domain = source_domain
        self.fill_mask = fill_mask
        self.repeat = repeat
        self.transform = self.get_transform(is_train)

        self.data_dict, _ = self.get_dict(infos)
        self.id_list = list(self.data_dict.keys())

        if is_train:
            self.train_list = random.sample(self.id_list, int(len(self.id_list) * 0.9))
            self.valid_list = random.sample(self.train_list, int(len(self.train_list) * 0.1))
            self.test_list = list(set(self.id_list).difference(set(self.train_list)))
            self.id_list = self.train_list
        elif (is_train is False and data_list is not None):
            self.id_list = data_list

        self.num_data = len(self.id_list)

    def __getitem__(self, index):

        def get_info_dict(index):
            index = index // self.repeat
            id = self.id_list[index]
            current_input_dir = dict()
            images = self.data_dict[id]['images']
            masks  = self.data_dict[id]['masks']
            for k in self.view_num:
                if self.single_frame:
                    if (k in images.keys() and k in masks.keys()):
                        if (images[k] is not None and masks[k] is not None):
                            image_list = np.array(nib.load(images[k]).dataobj)
                            mask_list  = np.array(nib.load(masks[k]).dataobj)
                            select_images_, select_masks_, mask_index = self.input_select(image_list, mask_list)
                            if mask_index == None:
                                continue
                            if np.sum(select_masks_) < 100:
                                continue
                            else:
                                current_input_dir[k] = {'images':select_images_, 'masks':select_masks_}
                else:
                    if k in images.keys():
                        if (images[k] is not None and masks[k] is not None):
                            image_list = np.array(nib.load(images[k]).dataobj)
                            if masks[k] is not None:
                                mask_list  = np.array(nib.load(masks[k]).dataobj)
                            else:
                                mask_list  = torch.zeros(image_list.shape)
                            
                            if len(image_list.shape) == 3:
                                video_length = image_list.shape[-1]
                                sample_rate = int(self.total_length/self.clip_length)
                                if video_length >= self.clip_length:
                                    if video_length < self.clip_length * sample_rate:
                                        sample_rate = video_length // self.clip_length
                                    start_index = random.randint(0, video_length - self.clip_length * sample_rate)
                                    end_index = start_index + self.clip_length
                                    select_images_ = image_list[:, :, start_index:end_index:sample_rate]
                                    select_masks_  = mask_list[:, :, start_index:end_index:sample_rate]
                                    mask_frames_ = np.sum(select_masks_, axis=(0,1))
                                    mask_index = np.where(mask_frames_ > 100, 1, 0)
                                    if self.fill_mask:
                                        select_masks_ = self.contour_to_mask(select_masks_)
                                    current_input_dir[k] = {'images':select_images_, 'masks':select_masks_}
                                elif video_length < self.clip_length:
                                    continue

                            elif len(image_list.shape) == 2:
                                continue

            while not current_input_dir:
                index = random.randint(0, self.num_data-1)
                current_input_dir, mask_index, index = get_info_dict(index)

            return current_input_dir, mask_index, index

        current_input_dir, mask_index, index = get_info_dict(index)

        if self.seg_parts:
            if self.view_num   == ['1']:
                BG = np.where(current_input_dir[self.view_num[0]]['masks'] == 0, 1, 0)
                LV = np.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = np.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                current_input_dir[self.view_num[0]]['masks'] = np.stack([BG, LV, RV], axis=0)
            elif self.view_num == ['2']:
                BG = np.where(current_input_dir[self.view_num[0]]['masks'] == 0, 1, 0)
                PA = np.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                current_input_dir[self.view_num[0]]['masks'] = np.stack([BG, PA], axis=0)
            elif self.view_num == ['3']:
                BG = np.where(current_input_dir[self.view_num[0]]['masks'] == 0, 1, 0)
                LV = np.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                RV = np.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                current_input_dir[self.view_num[0]]['masks'] = np.stack([BG, LV, RV], axis=0)
            elif self.view_num == ['4']:
                BG = np.where(current_input_dir[self.view_num[0]]['masks'] == 0, 1, 0)
                LV = np.where(current_input_dir[self.view_num[0]]['masks'] == 1, 1, 0)
                LA = np.where(current_input_dir[self.view_num[0]]['masks'] == 2, 1, 0)
                RA = np.where(current_input_dir[self.view_num[0]]['masks'] == 3, 1, 0)
                RV = np.where(current_input_dir[self.view_num[0]]['masks'] == 4, 1, 0)
                current_input_dir[self.view_num[0]]['masks'] = np.stack([BG, LV, LA, RA, RV], axis=0)
        else:
            current_input_dir[self.view_num[0]]['masks'] = np.where(current_input_dir[self.view_num[0]]['masks'] > 0, 1, 0)
        
        current_input_dir[self.view_num[0]] = self.transform(current_input_dir[self.view_num[0]])
        
        return current_input_dir[self.view_num[0]]['images'] / 255.0, current_input_dir[self.view_num[0]]['masks'], mask_index, index

    def __len__(self):
        if self.is_train:
            return self.num_data * self.repeat
        else:
            return self.num_data

    def is_number(self, s):
        try:
            float(s)
            return True
        except ValueError:
            pass
     
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
     
        return False

    def get_dict(self, infos):

        selected_dict = dict()

        for k, v in infos.items():
            if v['dataset_name'] in self.set_select:
                selected_dict.update({k:{'images':None, 'masks':None}})
                selected_dict[k]['images'] = v['views_images']
                selected_dict[k]['masks']  = v['views_labels']

        return selected_dict, None

    def input_select(self, images, masks):
        if len(masks.shape) == 3:
            mask_frames_ = np.sum(masks, axis=(0,1))
            mask_frames_ = np.where(mask_frames_ > 100, 1, 0)
            mask_frames_ = np.argwhere(mask_frames_ == 1)

            if mask_frames_.size == 0:
                return None, None, None

            select_index = random.choice(mask_frames_)[0]
            
            if self.single_frame:
                return images[:, :, select_index], masks[:, :, select_index], select_index
            else:
                if masks.shape[-1] == 3:
                    images = np.tile(images[:, :, 1:2],(1,1,self.clip_length))
                    masks  = np.tile(masks[:, :, 1:2], (1,1,self.clip_length))
                else:
                    r_index = random.randint(0, select_index if select_index < self.clip_length-1 else self.clip_length-1)
                    start = select_index - r_index
                    end = start + self.clip_length - 1
                    images = images[:, :, start:end]
                    masks  = masks[:, :, start:end]
                    select_index = r_index

                return images, masks, np.array([select_index])
        else:
            if self.single_frame:
                return images, masks, 0
            else:
                return np.tile(images,(self.clip_length,1,1)).transpose(1,2,0), np.tile(masks,(self.clip_length,1,1)).transpose(1,2,0), 0

    def contour_to_mask(self, input):
        organ_num = {'1':2,'2':1,'3':2,'4':4}
        masks = list()
        all_cls = list(set(list(input.reshape(-1))))
        all_cls.remove(0)

        h, w, _ = input.shape
        clip_length = input.shape[-1]
        for i in range(clip_length):
            contour = input[:,:,i]
            mask = np.zeros((h,w))
            for cls in range(1,organ_num[self.view_num[0]]+1):
                if cls>len(all_cls):
                    break
                contour_xy = np.argwhere(contour==all_cls[cls-1])
                img = np.zeros((h, w, 3), np.uint8)
                if list(contour_xy)!=[]:
                    cv2.fillPoly(img, [contour_xy], (255, 255, 255))
                    mask_xy = np.argwhere(img[:,:,0]==255)
                    for idx in mask_xy:
                        mask[idx[1],idx[0]] = cls
            mask = np.expand_dims(mask, axis=-1)
            masks.append(mask)
        return np.concatenate(masks, axis=-1)

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
    pass