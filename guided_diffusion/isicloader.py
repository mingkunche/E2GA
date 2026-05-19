import os
import sys
import pickle
import cv2
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import pandas as pd
from skimage.transform import rotate

class ISICDataset(Dataset):
    def __init__(self, args, data_path , transform = None, mode = 'Training',plane = False):


        df = pd.read_csv(os.path.join(data_path, 'ISBI2016_ISIC_Part1_' + mode + '_GroundTruth.csv'), encoding='gbk')
        self.name_list = df.iloc[:,1].tolist()
        self.label_list = df.iloc[:,2].tolist()
        self.data_path = data_path
        self.mode = mode

        self.transform = transform

    def __len__(self):
        return len(self.name_list)

    def __getitem__(self, index):
        """Get the images"""
        name = self.name_list[index]
        img_path = os.path.join(self.data_path, name)
        
        mask_name = self.label_list[index]
        msk_path = os.path.join(self.data_path, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(msk_path).convert('L')

        # if self.mode == 'Training':
        #     label = 0 if self.label_list[index] == 'benign' else 1
        # else:
        #     label = int(self.label_list[index])

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)


        return (img, mask, name)#读取并预变换后的图像，读取并预变换后的分割图，对应的图像文件名


class ISICDataset1(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.indice = [fname.split('_')[0] for fname in os.listdir(self.image_dir) if '.jpg' in fname]

        self.indice_list = sorted(self.indice,key= int)
        self.transform = transform

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, index):
        """Get the images"""
        indice = self.indice_list[index]
        img_name = indice+'_adversarial.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = indice+'_output_ens.jpg'
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, indice)


class ISICDataset2(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = os.path.join(data_dir, '输入的I0')
        self.mask_dir = os.path.join(data_dir,'正常的输出')
        self.indice = [fname.split('_')[-1].split('.')[0] for fname in os.listdir(self.image_dir) if '.jpg' in fname]
        self.indice_list = sorted(self.indice, key=int)
        self.transform = transform

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, index):
        """Get the images"""
        indice = self.indice_list[index]
        img_name = 'ISIC_'+indice +'.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = indice + '_output_ens.jpg'
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, indice)

class ISICDataset3(Dataset):
    def __init__(self, data_dir, transform=None):
        self.image_dir = data_dir
        
        self.indice = [fname.split('_')[-1].split('.')[0] for fname in os.listdir(self.image_dir) if '.jpg' in fname]
        self.indice_list = sorted(self.indice, key=int)
        self.transform = transform

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, index):
        """Get the images"""
        indice = self.indice_list[index]
        img_name = 'ISIC_'+indice +'.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        

        img = Image.open(img_path).convert('RGB')
        

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            

        return (img, indice)

class ISICDataset4(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.indice = [fname.split('_')[-1].split('.')[0] for fname in os.listdir(self.image_dir) if '.jpg' in fname]
        self.indice_list = sorted(self.indice,key= int)
        self.transform = transform

    def __len__(self):
        return len(self.indice_list)

    def __getitem__(self, index):
        """Get the images"""
        indice = self.indice_list[index]
        img_name = 'ISIC_' + indice + '.jpg'
        img_path = os.path.join(self.image_dir, img_name)
        mask_name = 'ISIC_' + indice + '_Segmentation'+'.png'
        mask_path = os.path.join(self.mask_dir, mask_name)

        img = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')

        if self.transform:
            state = torch.get_rng_state()
            img = self.transform(img)
            torch.set_rng_state(state)
            mask = self.transform(mask)

        return (img, mask, indice)
