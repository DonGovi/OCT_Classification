
import os
import cv2
import pandas as pd
import numpy as np
from skimage.util import random_noise

import torch
import torch.utils.data as data

def img_preprocess(img, argument=True):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)   # (h, w)
    if argument:
        gray = random_noise(gray, mode="gaussian", clip=True)
    
    return gray
        
    
class octDataset(data.Dataset):
    def __init__(self, data_path, data_list, label_file, argument=True):
        super(octDataset, self).__init__()
        self.path = data_path
        self.flist = data_list
        self.labelDF = pd.read_csv(label_file)
        self.argu = argument
        
    def __getitem__(self, idx):
        
        fname = self.flist[idx]
        ext_name = fname.split(".")[0]
        img = cv2.imread(os.path.join(self.path, fname))
        img = img_preprocess(img, self.argu)
        cls = self.labelDF[self.labelDF["id"] == ext_name].iloc[0, 1]
        if cls == 3:    # combine type 1 and 2 into a single class
            cls = 2
        
        sample = {"id":ext_name, "data":img, "label":cls}
        
        return sample
    
    def __len__(self):
        return len(self.flist)
    
    def collate_fn(self, batch):
        
        #ids = [x["id"] for x in batch]
        imgs = [x["data"] for x in batch]
        clss = [x["label"] for x in batch]
        
        max_h = max([img.shape[0] for img in imgs])
        max_w = max([img.shape[1] for img in imgs])
        num_imgs = len(imgs)
        inputs = torch.zeros(num_imgs, 1, max_h, max_w)
        
        labels = np.zeros(num_imgs, dtype=np.int64)
        #print(type(labels))
        for i in range(num_imgs):
            img = imgs[i]      #(h, w)
            im_h, im_w = img.shape
            inputs[i, 0, :im_h, :im_w] = torch.from_numpy(img)
            labels[i] = clss[i]
        
        targets = torch.from_numpy(labels.copy())
        return inputs, targets


if __name__ == "__main__":
    data_path = "F:/OCT/classification/non_stream/data/"
    label_file = "F:/OCT/classification/non_stream/ns_label.csv"
    dataset = octDataset(data_path, label_file)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=1, collate_fn=dataset.collate_fn)
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(targets, inputs.size())