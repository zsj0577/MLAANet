import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Qt5Agg')
import torchvision.transforms.functional as TF
from albumentations.augmentations import transforms as A
from albumentations.core.composition import Compose, OneOf

from albumentations.pytorch.transforms import ToTensorV2


class GlasDataset(Dataset):
    def __init__(self, images_path):

        self.images_path = images_path
        self.n_samples = len(images_path)

        self.image_transforms = transforms.Compose([
            transforms.ToPILImage(),
            # transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.1, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0,
                std=1,
            )])

        self.mask_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=0,
                std=1,
            )])

    def transform(self, image, mask):
        # Resize
        # resize = transforms.Resize(size=(512, 512))
        resize = transforms.Resize(size=(256, 256))
        image = resize(image)
        mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        # Transform to tensor
        image = torch.tensor(np.array(image))
        mask = torch.tensor(np.array(mask))
        return image, mask

    def __getitem__(self, index):
        """ Reading image """
        image = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        image = image/255.0
        image = np.transpose(image, (2, 0, 1))
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        image = self.image_transforms(image)

        """ Reading mask """
        mask = cv2.imread(self.images_path[index].replace(
            '.bmp', '_anno.bmp').replace('imgs', 'masks'), cv2.IMREAD_COLOR)
        # mask = cv2.imread(self.images_path[index], cv2.IMREAD_COLOR)
        mask = np.transpose(mask, (2, 0, 1))
        mask = mask[0,:,:]
        # plt.imshow(mask)
        # mask[mask == 0] = 0
        # mask[mask == 1] = 204 #mask[mask != 0] = 255
        # mask[mask == 2] = 255
        # mask = mask/255.0
        # mask = np.expand_dims(mask, axis=0)
        mask = mask.astype(np.float32)
        mask = torch.from_numpy(mask)
        mask = self.mask_transforms(mask)
        # mask = mask>0

        """ Transforms """
        image, mask = self.transform(image, mask)
        mask  = (mask>0) +0.0
        return image, mask

    def __len__(self):
        return self.n_samples
