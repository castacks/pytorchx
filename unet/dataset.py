from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from skimage import io, transform
import torch
import torch.nn as nn
import os
import json
import pycocotools.mask as pymask
import numpy as np
import tqdm


class XPlaneDataset(Dataset):
    def __init__(self, data_folder, transform=None):
        self.data_folder = data_folder
        self.annotations_json_file = os.path.join(data_folder, 'annotations_seg_bb.json')
        self.cached_indices = dict()
        print("Caching dataset indices")
        idx = 0
        with open(self.annotations_json_file, "r") as json_file:
            self.data = json.load(json_file)
            self.annotations = self.data['annotations']
            for i, seq in tqdm.tqdm(enumerate(self.annotations)):
                for j, frame in enumerate(seq):
                    self.cached_indices[idx] = [i, j]
                    idx += 1
        self.dataset_length = idx
        self.transform = transform
    

    def __len__(self):
        return self.dataset_length
    

    def __getitem__(self, idx):
        i, j = self.cached_indices[idx]
        frame = self.data['annotations'][i][j]
        image_file = os.path.join(self.data_folder, frame['image'])
        image = io.imread(image_file)

        mask = np.zeros(image.shape[:2]).astype('uint8')
        for label in frame['label']:
            mask += pymask.decode(label['segmentation'])
        
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return [image, mask]