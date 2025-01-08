import torch
from torch.utils import data
import torchvision.transforms as transforms
import cv2
import numpy as np
import pandas as pd

class myDatasets(data.Dataset):
    def __init__(self,seg_path="train_new.csv",unlabel_txt="unlabel.txt",
                 split='train',size=(224, 224),data_path='../Component 20/'):
        with open(unlabel_txt,"r") as f:
            unlabel=f.readlines()
            f.close()
        self.unlabel={}
        for item in unlabel:
            item=item[:-1]
            label=item.split('/')[0]
            if label not in self.unlabel.keys():
                self.unlabel[label]=[]
            self.unlabel[label].append(item)

        self.key=list(self.unlabel.keys())
        data = pd.read_csv(seg_path)
        self.data = np.array(data)
        self.data_path=data_path
        self.split=split
        self.size = size

        self.label={}
        for item in self.data:
            item=item[0]
            label=item.split('/')[2]
            if label not in self.label.keys():
                self.label[label]=[]
            self.label[label].append(item)

        immean = [0.5, 0.5, 0.5]  # RGB channel mean for imagenet
        imstd = [0.5, 0.5, 0.5]

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(immean, imstd),
        ])
    def __len__(self):
        return len(self.data)

    def process_seg(self,path):
        img = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img, -1)
        img = cv2.resize(img, self.size)
        character = img[:, :, -1]
        component = img[:, :, :-1]
        component1 = cv2.cvtColor(component, cv2.COLOR_BGR2GRAY)
        _, component = cv2.threshold(component1, 60, 255, cv2.THRESH_BINARY)

        component[component> 0] = 1


        character = np.stack((character,) * 3, axis=-1)
        component1= np.stack((component1,) * 3, axis=-1)

        return self.transform(character), component,self.transform(component1)

    def process_ret(self,path,m='character'):
        img = np.fromfile(path, dtype=np.uint8)
        img = cv2.imdecode(img, -1)
        img = cv2.resize(img, self.size)
        if m=='character':
            img = img[:, :, -1]
            img = np.stack((img,) * 3, axis=-1)
        else:
            img = img[:, :, :-1]
        return self.transform(img)

    def get_negtive(self,name):
        neg=name
        while(neg==name):
            neg=np.random.choice(self.key)
        return neg

    def __getitem__(self, idx):
        path,_,CDC,CC,IC,_,label=self.data[idx]
        # data_path, _ , connected domain count, circle count, intersection count, _, label
        name=path.split("/")
        path=name[2]

        for i in name[3:]:
            path+='/'
            path+=i

        #rescale the count to category
        if CDC > 2:
            CDC = 2
        elif CDC == 2:CDC = 1
        else:CDC = 0
        if CC > 1:CC = 2
        if IC > 4:IC = 4

        character,seg_label,component=self.process_seg(self.data_path+path)

        if self.split=='test':
            return character,seg_label,label,CDC,CC,IC,component
        pos=name[2]
        neg=self.get_negtive(pos)
        pos=self.process_ret(self.data_path+np.random.choice(self.label[pos]),m='component')
        neg=self.process_ret(self.data_path+np.random.choice(self.unlabel[neg]),m='character')

        return character,seg_label,label,CDC,CC,IC,pos,neg,component
