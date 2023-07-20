# -*- coding: UTF-8 -*-
"""
@version: 1.0
@PackageName: hs_code - main.py
@author: yonghao
@Description: 训练与验证过程
@since 2021/11/16 15:38
"""
import os
import torch
from torch import nn
from run import Train, Val
from pcg_dataset import PCGDataset
from torch.utils.data import DataLoader
import numpy as np

from models import model3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == '__main__':

    base_path = r'D:\Pcg_code\PCG_DATA_CUT'
    result_dir_path = r'D:\Pcg_code\result'  # 结果信息保存路径
    folds = 10  # 交叉验证折数
    fold_acc = []
    fold_f1 = []

    for f in range(folds):
        model = model3.QCnn(2)  # pytorch 模型 in [batch,n_mfcc,T]  out [batch,class_num] --- class_num=2

        train_db = PCGDataset(root=base_path, fold_path=os.path.join(base_path, f'fold_dataset_{f}'), train=True)
        val_db = PCGDataset(root=base_path, fold_path=os.path.join(base_path, f'fold_dataset_{f}'), train=False)
        train_loader = DataLoader(train_db, batch_size=512, shuffle=True, num_workers=0, drop_last=True)
        val_loader = DataLoader(val_db, batch_size=512, shuffle=True, num_workers=0)

        val = Val(val_loader, device=device, result_dir_path=os.path.join(result_dir_path, str(f)))

        train = Train(epochs=40, lr=2e-4, loss=nn.CrossEntropyLoss(), weight_decay=0.01,
              result_dir_path=os.path.join(result_dir_path, str(f)),
              model=model, device=device, train_dataloader=train_loader, val=val, n_folds=folds)

        train.run()

    for j in range(folds):
        path = os.path.join(result_dir_path, f'{j}/val_index.csv')
        f1 = []
        acc = []
        with open(path, 'r') as f:
            data = f.read()
        for i in data.split('\n'):  # 这里注意要判空
            if len(i) == 0:
                continue
            else:
                acc.append(i.split(',')[0])
                f1.append(i.split(',')[-5])

        fold_acc.append(max(acc))
        fold_f1.append(f1[acc.index(max(acc))])

    print('acc: ', sum(fold_acc)/len(fold_acc), 'f1: ', sum(fold_f1)/len(fold_f1))  # 快速求list的均值
