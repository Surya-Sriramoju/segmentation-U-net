import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import numpy as np
import albumentations as A
import cv2

class CityscapesDataset(Dataset):
    def __init__(self, split, root_dir, target_type='semantic', mode='fine', transform=None, eval=False):
        self.transform = transform
        if mode == 'fine':
            self.mode = 'gtFine'
        
        elif mode == 'coarse':
            self.mode = 'gtCoarse'
        
        self.split = split
        self.yLabel_list = []
        self.XImg_list = []
        self.eval = eval
        self.remap = {0:255, 1:255,2:255,3:255,4:255,5:255,6:255, 7:0, 8:1, 9:255, 10:255, 11:2, 12:3, 13:4, 14:255, 15:255,
                      16:255, 17:5, 18:255, 19:6, 20:7, 21:8, 22:9, 23:10, 24:11, 25:12, 26:13, 27:14, 28:15, 29:255, 30:255,
                      31:16, 32:17, 33:18}

        self.label_path = os.path.join(os.getcwd(), root_dir+'/'+self.mode+'/'+self.split)
        self.rgb_path = os.path.join(os.getcwd(), root_dir+'/leftImg8bit/'+self.split)
        city_list = sorted(os.listdir(self.label_path))
        for city in city_list:
            temp = os.listdir(self.label_path+'/'+city)
            list_items = temp.copy()
    
            # 19-class label items being filtered
            for item in temp:
                if not item.endswith('labelIds.png', 0, len(item)):
                    list_items.remove(item)

            # defining paths
            list_items = ['/'+city+'/'+path for path in list_items]

            self.yLabel_list.extend(sorted(list_items))
            self.XImg_list.extend(
                ['/'+city+'/'+path for path in sorted(os.listdir(self.rgb_path+'/'+city))]
            )

    def __len__(self):
        length = len(self.XImg_list)
        return length
    def remap_labels(self, tensor):
        for old_label_id, new_label_id in self.remap.items():
            tensor[tensor==old_label_id] = new_label_id
        return tensor

    def __getitem__(self, index):
        image = Image.open(self.rgb_path+self.XImg_list[index])
        label = Image.open(self.label_path+self.yLabel_list[index])

        if self.transform is not None:
            transformed=self.transform(image=np.array(image), mask=np.array(label))
            image = transformed["image"]
            label = transformed["mask"]

        label = self.remap_labels(label)
        label = label.clamp(max = 18)
        # label = torch.nn.functional.interpolate(label.unsqueeze(0).unsqueeze(0), size=(128,256), mode='nearest')
        # label = label.squeeze(0).squeeze(0)
        label = label.type(torch.LongTensor)
        return image, label