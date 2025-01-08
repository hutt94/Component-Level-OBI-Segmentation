# -*- codeing = utf-8 -*-
# @Time : 2024/5/7 14:00
# @Author : 李昌杏
# @File : train.py
# @Software : PyCharm
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from network.model import Character_encoder,Component_encoder,Baseline_encoder
from datasets import myDatasets

class Option:
    def __init__(self):
        parser = argparse.ArgumentParser(description="args for model")
        # dataset
        parser.add_argument('--seg_path', type=str, default=f"test_new.csv")
        parser.add_argument('--data_path', type=str, default=f"../Component 20/")
        parser.add_argument('--unlabel_txt', type=str, default=f"unlabel.txt")
        parser.add_argument("--seed", default=1234)
        #train
        parser.add_argument("--img_size", default=224)
        parser.add_argument("--epoch", default=200)
        parser.add_argument("--warmup_epochs", default=3)
        parser.add_argument("--batch_size", default=8)
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
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    #<<<<<<<<<<<<<<<<datasets<<<<<<<<<<<<<<<<<<<<
    datasets_valid=myDatasets(args.seg_path,args.unlabel_txt,split='test',data_path=args.data_path)
    valid_loader = DataLoader(datasets_valid, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    #<<<<<<<<<<<<<<<<models<<<<<<<<<<<<<<<<<<<<<<
    model_component= Component_encoder(args.num_class,args.PVT_pre_weight).cuda()
    model_component.load_state_dict(torch.load(f'weights/componet_seg_best.pth'))
    model_character= Character_encoder(args.num_class,args.PVT_pre_weight).cuda()
    model_character.load_state_dict(torch.load(f'weights/character_seg_best.pth'))
    model_base = Baseline_encoder(args.num_class).cuda()
    model_base.load_state_dict(torch.load(f'weights/baseline_best.pth'))

    model_character.eval()
    model_component.eval()
    with torch.no_grad():
        acc=0
        IC_acc=0
        CC_acc=0
        CDC_acc=0
        model_character.eval()
        model_component.eval()

        pred = []
        target = []
        with torch.no_grad():
            for i, item in enumerate(valid_loader):
                character,seg_label,label,CDC,CC,IC,component = item
                character = character.cuda()
                component = component.cuda()
                seg_label = seg_label.cuda()
                label = label.cuda()
                CDC, CC, IC = CDC.cuda(), CC.cuda(), IC.cuda()

                outputs_seg, cls = model_character(character)
                seg_com = torch.argmax(outputs_seg, dim=1).unsqueeze(1).float()
                seg_com = seg_com.repeat(1, 3, 1, 1)

                cls,out_IC, out_CDC, out_CC = model_base(seg_com)

                pred_seg = torch.argmax(outputs_seg, dim=1).detach().cpu().numpy()
                for item in pred_seg:
                    pred.append(item)
                for item in seg_label.detach().cpu().numpy():
                    target.append(item)

                acc+=(((torch.argmax(cls,dim=1))==label).sum().item())
                IC_acc+=(((torch.argmax(out_IC,dim=1))==IC).sum().item())
                CC_acc+=(((torch.argmax(out_CC,dim=1))==CC).sum().item())
                CDC_acc+=(((torch.argmax(out_CDC,dim=1))==CDC).sum().item())

        pred = np.array(pred)
        target = np.array(target)
        print(pred.shape,target.shape)
        acc/=datasets_valid.__len__()
        compute_metrics(pred,target,2)
        IC_acc/=datasets_valid.__len__()
        CDC_acc/=datasets_valid.__len__()
        CC_acc/=datasets_valid.__len__()
        print(f"acc={acc}\n",
              f"IC_acc={IC_acc}\n",
              f"CC_acc={CC_acc}\n",
              f"CDC_acc={CDC_acc}")


def compute_metrics(pred, label, num_classes):
    # 初始化变量
    iou_list = []
    precision_list = []
    recall_list = []

    for c in range(num_classes):
        # 计算TP, FP, FN, TN
        tp = np.sum((pred == c) & (label == c))
        fp = np.sum((pred == c) & (label != c))
        fn = np.sum((pred != c) & (label == c))

        # 计算IOU, Precision, Recall
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else float('nan')
        precision = tp / (tp + fp) if (tp + fp) > 0 else float('nan')
        recall = tp / (tp + fn) if (tp + fn) > 0 else float('nan')

        # 将结果加入列表
        iou_list.append(iou)
        precision_list.append(precision)
        recall_list.append(recall)

    # 计算mIOU, mean Precision 和 mean Recall
    mIOU = np.nanmean(iou_list)
    mean_precision = np.nanmean(precision_list)
    mean_recall = np.nanmean(recall_list)

    print(f"mIOU:{mIOU}\nclass iou {iou_list}")
    print(f"prec:{mean_precision}\nclass prec {precision_list}")
    print(f"recall:{mean_recall}\nclass recall {recall_list}")


# mIOU:0.6145021431433211
# class iou [0.9738700128589058, 0.25513427342773637]
# prec:0.6611499332235472
# class prec [0.9918575476028204, 0.33044231884427394]
# recall:0.7549543382945267
# class recall [0.9817186447065603, 0.5281900318824931]
# acc=0.33905579399141633
#  IC_acc=0.4034334763948498
#  CC_acc=0.6695278969957081
#  CDC_acc=0.6223175965665236



#wo no
# mIOU:0.6032271486976477
# class iou [0.9657095503508308, 0.24074474704446466]
# prec:0.6360443114097261
# class prec [0.9937185489251004, 0.27837007389435187]
# recall:0.8060382988880583
# class recall [0.9716408364160686, 0.6404357613600481]
# acc=0.33047210300429186
#  IC_acc=0.41201716738197425
#  CC_acc=0.6952789699570815
#  CDC_acc=0.6394849785407726

#wo 3 aux
# mIOU:0.5920496252782944
# class iou [0.9633799795747867, 0.22071927098180213]
# prec:0.6249814350752168
# class prec [0.9931990590536329, 0.2567638110968007]
# recall:0.7905098806836949
# class recall [0.9697773627445505, 0.6112423986228392]
# acc=0.24034334763948498
#  IC_acc=0.33047210300429186
#  CC_acc=0.6652360515021459
#  CDC_acc=0.6266094420600858

if __name__ == '__main__':
    main()
