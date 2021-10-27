import logging
from os import listdir
from os.path import splitext
from pathlib import Path
import matplotlib.pyplot as plt

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class BasicDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        
        self._images_dir = images_dir
        self._masks_dir = masks_dir
        
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        self.ids = [splitext(file)[0] for file in listdir(images_dir) if not file.startswith('.')]
        if not self.ids:
            raise RuntimeError(f'No input file found in {images_dir}, make sure you put your images there')
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH))
        img_ndarray = np.asarray(pil_img)

        if img_ndarray.ndim == 2 and not is_mask:
            img_ndarray = img_ndarray[np.newaxis, ...]
        elif not is_mask:
            img_ndarray = img_ndarray.transpose((2, 0, 1))

        if not is_mask:
            img_ndarray = img_ndarray / 255

        return img_ndarray

    @classmethod
    def load(cls, filename):
        ext = splitext(filename)[1]
        if ext in ['.npz', '.npy']:
            return Image.fromarray(np.load(filename))
        elif ext in ['.pt', '.pth']:
            return Image.fromarray(torch.load(filename).numpy())
        else:
            return Image.open(filename)
        
    def get(self):
         return self.__getitem__(0)

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_img = str(self._masks_dir) + "/" + str(name) + "_mask.jpg"
        orig_img = str(self._images_dir) + "/" + str(name) + ".jpg"
        
        mask = self.load(mask_img)
        img = self.load(orig_img)
        
        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)
        
        mask_tensor = torch.as_tensor(mask.copy()).long().contiguous()
        
        mask_tensor[mask_tensor<=0] = 256
        min_t = torch.min(mask_tensor)
        mask_tensor[mask_tensor>=256] = 0
        
        max_t = torch.max(mask_tensor) - min_t -5
        mask_tensor[mask_tensor<max_t] = 0
        mask_tensor[mask_tensor>=max_t] = 1
        
        result = {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': mask_tensor
        }
        return result

class ClothingDataset(BasicDataset):
    def __init__(self, images_dir, masks_dir, scale=1):
        super().__init__(images_dir, masks_dir, scale, mask_suffix='_mask')
