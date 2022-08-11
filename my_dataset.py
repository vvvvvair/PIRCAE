from PIL import Image
import torch
from torch.utils.data import Dataset
import numpy as np


class MyDataSet(Dataset):

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img1 = Image.open(self.images_path[item])
        img2 = Image.open(self.images_path[item].replace("bands","ints"))
        
        box1 = (0, 0, 24, 48)
        box2=  (24,0,48,48)
        crop_img1 = img1.crop(box1)
        crop_img2 = img2.crop(box2)
        img3 = Image.new(img1.mode, (48, 48))
        img3.paste(crop_img1, box=(0, 0))
        img3.paste(crop_img2, box=(24, 0))
        img = np.dstack([img1,img2,img3])
       
      
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):

        images, labels = tuple(zip(*batch))
        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels

