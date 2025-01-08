# -*- codeing = utf-8 -*-
# @Time : 2024/5/7 14:00
# @Author : 李昌杏
# @File : train.py
# @Software : PyCharm
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from network.model import Baseline_encoder
from datasets import myDatasets
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--seg_path', type=str, default=f"datalist/train_new.csv")
        parser.add_argument('--seg_path_test', type=str, default=f"datalist/test_new.csv")
        parser.add_argument('--data_path', type=str, default=f"../Component 20/")
        parser.add_argument('--unlabel_txt', type=str, default=f"datalist/unlabel.txt")
        parser.add_argument("--seed", default=1234)
        #train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=200)
        parser.add_argument("--warmup_epochs", default=3)
        parser.add_argument("--batch_size", default=32)
        parser.add_argument("--lr", default=0.001)
        parser.add_argument("--weight_decay", default= 0.04)
        #net
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--PVT_pre_weight", default='pvt_medium.pth')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def main():
    args=Option().parse()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #<<<<<<<<<<<<<<<<datasets<<<<<<<<<<<<<<<<<<<<
    datasets=myDatasets(args.seg_path,args.unlabel_txt,split='train',data_path=args.data_path)
    datasets_valid=myDatasets(args.seg_path_test,args.unlabel_txt,split='test',data_path=args.data_path)
    train_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    valid_loader = DataLoader(datasets_valid, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=4)
    #<<<<<<<<<<<<<<<<models<<<<<<<<<<<<<<<<<<<<<<
    model = Baseline_encoder(args.num_class).cuda()
    #<<<<<<<<<<<<<<<<loss_initial<<<<<<<<<<<<<<<<
    loss_cn = torch.nn.CrossEntropyLoss().cuda()
    #<<<<<<<<<<<<<<<<optimize<<<<<<<<<<<<<<<<<<<<
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer = torch.optim.Adam(model.parameters(), lr)

    valid_cls_min = -1
    for epoch in range(args.epoch):
        epoch_train_contrastive_loss = 0
        model.train()
        sum_loss = 0

        for batch_idx, data in enumerate(tqdm(train_loader)):

            character_tri,character,seg_label,label,CDC,CC,IC,pos,neg,component=data

            component=component.cuda()
            label=label.cuda()
            CDC, CC, IC=CDC.cuda(),CC.cuda(),IC.cuda()

            cls ,out_IC,out_CDC,out_CC=model(component)

            IC_loss = loss_cn(out_IC,IC)
            CC_loss = loss_cn(out_CC,CC)
            CDC_loss = loss_cn(out_CDC,CDC)
            cls_loss = loss_cn(cls,label)

            loss = cls_loss + IC_loss + CC_loss + CDC_loss

            epoch_train_contrastive_loss += loss.item()

            loss.backward()
            sum_loss += loss.item()
            optimizer.step()
            optimizer.zero_grad()


        acc=0
        IC_acc=0
        CC_acc=0
        CDC_acc=0
        model.eval()
        with torch.no_grad():
            for i, item in enumerate(valid_loader):
                character,seg_label,label,CDC,CC,IC,component = item
                component = component.cuda()
                label = label.cuda()
                CDC, CC, IC = CDC.cuda(), CC.cuda(), IC.cuda()

                cls ,out_IC, out_CDC, out_CC = model(component)

                acc += (((torch.argmax(cls, dim=1)) == label).sum().item())
                IC_acc += (((torch.argmax(out_IC, dim=1)) == IC).sum().item())
                CC_acc += (((torch.argmax(out_CC, dim=1)) == CC).sum().item())
                CDC_acc += (((torch.argmax(out_CDC, dim=1)) == CDC).sum().item())

            acc /= datasets_valid.__len__()
            IC_acc /= datasets_valid.__len__()
            CDC_acc /= datasets_valid.__len__()
            CC_acc /= datasets_valid.__len__()
        print(f"{sum_loss/len(train_loader)}\n"
              f"acc={acc}\n",
              f"IC_acc={IC_acc}\n",
              f"CC_acc={CC_acc}\n",
              f"CDC_acc={CDC_acc}")

        if acc > valid_cls_min:
            valid_cls_min = acc
            torch.save(model.state_dict(), f'weights/baseline_best.pth')

# acc=0.8111587982832618
#  IC_acc=0.6995708154506438
#  CC_acc=0.8326180257510729
#  CDC_acc=0.8497854077253219
if __name__ == '__main__':
    main()
