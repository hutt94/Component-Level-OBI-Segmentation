# -*- codeing = utf-8 -*-
# @Time : 2024/3/15 13:36
# @Author : 李昌杏
# @File : visual.py
# @Software : PyCharm

import cv2
import matplotlib.pyplot as plt
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader
from network.model import Character_encoder
from datasets import myDatasets

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--seg_path', type=str, default=f"datalist/test_new.csv")
        parser.add_argument('--data_path', type=str, default=f"../Component 20/")
        parser.add_argument('--unlabel_txt', type=str, default=f"datalist/unlabel.txt")
        parser.add_argument("--seed", default=1234)
        #train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=200)
        parser.add_argument("--warmup_epochs", default=3)
        parser.add_argument("--batch_size", default=16)
        parser.add_argument("--lr", default=3e-3)
        parser.add_argument("--min_lr", default=1e-4)
        parser.add_argument("--weight_decay", default= 0.04)
        parser.add_argument("--weight_decay_end", default=0.4)
        #net
        parser.add_argument("--m", default=1)
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--PVT_pre_weight", default='pvt_medium.pth')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def main():
    args=Option().parse()
    #<<<<<<<<<<<<<<<<datasets<<<<<<<<<<<<<<<<<<<<
    datasets_valid=myDatasets(args.seg_path,args.unlabel_txt,split='test',data_path=args.data_path)
    valid_loader = DataLoader(datasets_valid, batch_size=args.batch_size,  pin_memory=True,num_workers=4)
    #<<<<<<<<<<<<<<<<models<<<<<<<<<<<<<<<<<<<<<<
    model_character= Character_encoder(args.num_class,args.PVT_pre_weight).cuda()
    model_character.load_state_dict(torch.load(f'weights/temp/.ipynb_checkpoints/zi_seg_best.pth'))

    model_character.eval()
    index = 0
    with torch.no_grad():
            for i, item in enumerate(valid_loader):
                character,seg_label,label,connect,cycle,cross,component = item
                character = character.cuda()

                outputs_seg, cls = model_character(character)

                pred_seg = torch.argmax(outputs_seg, dim=1).detach().cpu().numpy()
                for idx,item in enumerate(pred_seg):
                    red = np.ones((224,224,3))*255
                    red[item>0]=[255,0,0]
                    plt.imshow(red)
                    plt.axis('off')
                    plt.savefig(f"output/our/{index}_{label[idx].item()}-1.png",bbox_inches = 'tight',pad_inches=0)

                    img = character[idx].cpu().permute(1,2,0).numpy()
                    img = cv2.bitwise_not(img)
                    img[item>0] = [255,0,0]
                    plt.imshow(img)
                    plt.axis('off')
                    plt.savefig(f"output/our/{index}_{label[idx].item()}.png",bbox_inches = 'tight',pad_inches=0)
        #
                    index +=1
                print(index)
main()
