# -*- codeing = utf-8 -*-
# @Time : 2024/5/7 14:00
# @Author : 李昌杏
# @File : train.py
# @Software : PyCharm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch import nn
import utils
from network.model import Character_encoder,Component_encoder
from datasets import myDatasets

class Component_loss(nn.Module):
    def __init__(self,margin=1):
        super(Component_loss, self).__init__()
        self.margin=margin

    def forward(self,a,p1,p2):
        dist1=torch.sqrt(torch.sum(torch.square(a-p1),dim=1))
        dist2=torch.sqrt(torch.sum(torch.square(a-p2),dim=1))

        basic_loss=torch.abs(dist1-dist2)-self.margin
        basic_loss=basic_loss.unsqueeze(dim=1)
        loss=torch.cat((basic_loss, torch.zeros_like(basic_loss)),dim=1).max(dim=1)
        return loss[0].mean()
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
        parser.add_argument("--batch_size", default=24)
        parser.add_argument("--lr", default=3e-3)
        parser.add_argument("--min_lr", default=1e-4)
        parser.add_argument("--weight_decay", default= 0.04)
        parser.add_argument("--weight_decay_end", default=0.4)
        parser.add_argument("--alpha", default=1)
        parser.add_argument("--beta", default=1)
        #net
        parser.add_argument("--m", default=1)
        parser.add_argument("--num_class", default=20)
        parser.add_argument("--PVT_pre_weight", default='pvt_medium.pth')
        self.parser = parser

    def parse(self):
        return self.parser.parse_args()

def main():
    # <<<<<<<<<<<<<<<<initial<<<<<<<<<<<<<<<<<<<<
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
    train_loader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=2)
    valid_loader = DataLoader(datasets_valid, batch_size=args.batch_size, shuffle=True, pin_memory=True,num_workers=2)
    #<<<<<<<<<<<<<<<<models<<<<<<<<<<<<<<<<<<<<<<
    model_component= Component_encoder(args.num_class,args.PVT_pre_weight).cuda()
    model_character= Character_encoder(args.num_class,args.PVT_pre_weight).cuda()
    #<<<<<<<<<<<<<<<<loss_initial<<<<<<<<<<<<<<<<
    loss_cn = torch.nn.CrossEntropyLoss().cuda()
    character_loss= nn.TripletMarginLoss(margin=1, p=2).cuda()
    component_loss= Component_loss(margin=args.m).cuda()
    dice_loss = utils.DiceLoss(2)
    #<<<<<<<<<<<<<<<<optimize<<<<<<<<<<<<<<<<<<<<
    lr = args.lr
    weight_decay = args.weight_decay
    optimizer_component = torch.optim.AdamW(model_component.parameters(), lr, weight_decay=weight_decay)
    optimizer_character = torch.optim.AdamW(model_character.parameters(), lr, weight_decay=weight_decay)
    #<<<<<<<<<<<<<<<<scheduler initial<<<<<<<<<<<
    lr_schedule = utils.cosine_scheduler(
        lr * (args.batch_size) / 256.,  # linear scaling rule
        args.min_lr,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,  # linear scaling rule
        args.weight_decay_end,
        args.epoch, len(train_loader),
        warmup_epochs=args.warmup_epochs,
        early_schedule_epochs=0,
    )
    fp16_scaler = torch.cuda.amp.GradScaler()

    valid_loss_min = -1

    for epoch in range(args.epoch):
        epoch_train_contrastive_loss = 0
        model_character.train()
        model_component.train()
        for batch_idx, data in enumerate(tqdm(train_loader)):

            it = (len(train_loader)) * epoch + batch_idx  # global training iteration
            for i, param_group in enumerate(optimizer_component.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]
            for i, param_group in enumerate(optimizer_character.param_groups):
                if i == 0 or i == 1:
                    param_group['lr'] = lr_schedule[it] * 0.1
                else:
                    param_group["lr"] = lr_schedule[it]
                if i == 0 or i == 2:  # only the first group is regularized; look at get_params_groups for details
                    param_group["weight_decay"] = wd_schedule[it]

            character,seg_label,label,CDC,CC,IC,pos,neg,component=data

            character=character.cuda()
            component=component.cuda()
            seg_label=seg_label.cuda()
            label=label.cuda()
            CDC, CC, IC=CDC.cuda(),CC.cuda(),IC.cuda()
            pos, neg=pos.cuda(),neg.cuda()

            with torch.cuda.amp.autocast(fp16_scaler is not None):
                optimizer_component.zero_grad()
                optimizer_character.zero_grad()

                # <<<<<<<<<<<<<<<<calculate loss_unan<<<<<<<<<<<<<<<<<<<<
                component_logits, component_fea=model_component(component,False)  # class_logits,cls_token,
                _, pos_fea= model_component(pos,False)  # class_logits,cls_token,
                _, neg_fea= model_character(neg,False) # class_logits,cls_token,
                character_logits, character_fea= model_character(character,False)  # class_logits,cls_token,

                com_loss = component_loss(character_fea,component_fea,pos_fea)
                cha_loss = character_loss(component_fea,character_fea,neg_fea)

                # classification may speed up the convergence
                cls_loss = (loss_cn(component_logits,label)+loss_cn(character_logits,label))/2

                loss_unan = cls_loss + 0.01 * com_loss + 100 * cha_loss

                # <<<<<<<<<<<<<<<<calculate loss_anno<<<<<<<<<<<<<<<<<<<<
                outputs_seg, cls = model_character(character)
                loss_ce = loss_cn(outputs_seg, seg_label[:].long())
                loss_dice = dice_loss(outputs_seg, seg_label, softmax=True)

                loss_anno = 0.4 * loss_ce + 0.6 * loss_dice + loss_cn(cls, label)

                #generate segmentation result of component
                seg_com = torch.argmax(outputs_seg, dim=1).unsqueeze(1).float()
                seg_com = seg_com.repeat(1, 3, 1, 1)

                # <<<<<<<<<<<<<<<<calculate loss_stroke<<<<<<<<<<<<<<<<<<<<
                out_IC,out_CDC,out_CC=model_component(seg_com)

                IC_loss = loss_cn(out_IC,IC)
                CC_loss = loss_cn(out_CC,CC)
                CDC_loss = loss_cn(out_CDC,CDC)

                loss_stroke = IC_loss + CC_loss + CDC_loss


                # <<<<<<<<<<<<<<<<overall loss<<<<<<<<<<<<<<<<<<<<
                loss = loss_anno + args.alpha * loss_unan + args.beta * loss_stroke

                epoch_train_contrastive_loss += loss.item()
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer_component)
                fp16_scaler.step(optimizer_character)
                fp16_scaler.update()

        # <<<<<<<<<<<<<<<<valid<<<<<<<<<<<<<<<<<<<<
        iou=[]
        acc=[]
        IC_acc=[]
        CC_acc=[]
        CDC_acc=[]
        model_character.eval()
        model_component.eval()
        with torch.no_grad():
            for i, item in enumerate(valid_loader):
                character,seg_label,label,CDC,CC,IC,component = item
                character = character.cuda()
                seg_label = seg_label.cuda()
                label = label.cuda()
                CDC, CC, IC = CDC.cuda(), CC.cuda(), IC.cuda()

                outputs_seg, cls = model_character(character)
                seg_com = torch.argmax(outputs_seg, dim=1).unsqueeze(1).float()
                seg_com = seg_com.repeat(1, 3, 1, 1)

                out_IC, out_CDC, out_CC = model_component(seg_com)

                iou.append(utils.mean_iou(torch.argmax(outputs_seg, dim=1), seg_label, 2))
                acc.append(((torch.argmax(cls,dim=1))==label).sum().item()/label.shape[0])
                IC_acc.append(((torch.argmax(out_IC,dim=1))==IC).sum().item()/label.shape[0])
                CC_acc.append(((torch.argmax(out_CC,dim=1))==CC).sum().item()/label.shape[0])
                CDC_acc.append(((torch.argmax(out_CDC,dim=1))==CDC).sum().item()/label.shape[0])


        acc=np.mean(acc)
        iou=np.mean(iou)
        IC_acc=np.mean(IC_acc)
        CDC_acc=np.mean(CDC_acc)
        CC_acc=np.mean(CC_acc)
        print(f"epoch:{epoch + 1}-tri_loss=>>{epoch_train_contrastive_loss /len(train_loader)}"
              f"miou={iou}--acc={acc}\n",
              f"IC_acc={IC_acc}\n",
              f"CC_acc={CC_acc}\n",
              f"CDC_acc={CDC_acc}")

        if iou>valid_loss_min:
            valid_loss_min=iou
            torch.save(model_component.state_dict(), f'weights/component_seg_best.pth')
            torch.save(model_character.state_dict(), f'weights/character_seg_best.pth')


if __name__ == '__main__':
    main()
