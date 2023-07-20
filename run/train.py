# -*- coding: UTF-8 -*-
"""
@version: 1.0
@PackageName: hs_model - train.py
@author: yonghao
@Description: 训练类
@since 2021/09/13 15:51
"""
import os, csv
import time
import math
import torch
from torch import optim


class Train:
    _epoch_loss_file_name = 'epoch_loss'
    _mini_batch_loss_file_name = 'mini_batch_loss'
    _train_index_file_name = 'train_index'
    _val_index_file_name = 'val_index'
    _model_file_name = 'hs_model'

    def __init__(self, epochs, lr, loss, weight_decay, result_dir_path,
                 model, device, train_dataloader, val, gamma=0.01, n_folds=10):
        """
        初始化训练过程所需的参数
        :param epochs: 迭代次数
        :param lr: 学习率
        :param loss: loss对象
        :param weight_decay: L2的正则化超参数
        :param result_dir_path: 结果数据保存文件夹路径
        :param model: 模型对象
        :param device: 装置对象(cpu or gpu)
        :param train_dataloader: 训练集加载器
        :param val: 验证集的验证对象
        :param gamma: 使用学习率衰减时指定的 衰减率
        :param n_folds: 当前所使用的交叉验证折数
        """
        self.epochs = epochs
        self.loss = loss.to(device)
        self.model = model.to(device)
        self.dataloader = train_dataloader
        # 不同数据的验证对象
        self.val = val
        self.device = device
        self.result_dir_path = result_dir_path

        # 构建反向传播学习优化器
        params = list(self.model.parameters())
        if self.loss.parameters():
            params.extend(self.loss.parameters())
        self.optimizer = optim.Adam(params=params, lr=lr, weight_decay=weight_decay)

        # 构建梯度衰减器
        # scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[20, 50, 100], gamma=0.1)
        if not gamma:
            lf = lambda x: ((1 + math.cos(x * math.pi / self.epochs)) / 2) * (1 - gamma) + gamma
            # self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=milestones, gamma=gamma)
            self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lf)
        self.n_folds = n_folds

    def run(self):
        # 绘制模型流程图
        # self.tb_writer.add_graph(self.models)
        # 如果已存在训练好的模型参数文件则不执行训练过程
        # if os.path.exists(os.path.join(self.result_dir_path, f'{self.__class__._model_file_name}.mdl')):
        #     return None

        acc = 0.  # 保存运行过程中的最高准确率值

        for epoch in range(self.epochs):
            # 开启训练模式（用于在训练过程中启用 dropout 与 batchNorm）
            self.model.train(mode=True)

            epoch_loss_sum = 0.  # 用于保存每次迭代的所有mini_batch的loss累加

            # mini-batch训练
            for idx, (data, labels, _) in enumerate(self.dataloader):
                # step1:将 数据+标签 加入gpu进行运算
                data, labels = data.to(self.device), labels.to(self.device)

                # step2: 模型运行生成逻辑值等数据
                # out 若为单个值则为预测逻辑值；若为多个值第一个值为预测逻辑值；第二个为特征编码值
                out = self.model(data)

                # step3: 使用loss函数反向传播
                loss = self.loss(out, labels)

                # 记录一个mini-batch的loss值
                loss_value = loss.item()

                self._record_mini_batch_loss(epoch + 1, idx + 1, loss_value)  # 在一个csv文件中保存loss的值

                # 记录loss值
                epoch_loss_sum += loss_value  # 记录单次迭代的loss值

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # 衰减学习率（按迭代次数）
            if hasattr(self, 'scheduler'):
                self.scheduler.step()

            # 记录 epoch loss值
            self._record_loss(epoch + 1, epoch_loss_sum / len(self.dataloader))

            # 使用val数据集对当前学习到的模型进行评估
            # val: acc, ppv, se, spe, f1_score
            val_result = self.val(self.model, self.__class__._val_index_file_name)
            acc_ = val_result[0]
            if acc_ >= acc:
                torch.save(self.model.state_dict(),
                           os.path.join(self.result_dir_path, f'{self.__class__._model_file_name}.mdl'))
                acc = acc_

    def load_trained_model(self, model):
        """
        加载训练过的模型
        :return:
        """
        model.load_state_dict(
            torch.load(os.path.join(self.result_dir_path, f'{self.__class__._model_file_name}.mdl')))

    def _record_mini_batch_loss(self, epoch, idx, loss_value):
        """
        记录一个mini_batch的训练Loss值
        :param epoch: 当前的迭代次数
        :param idx: 当前的 mini-batch 数
        :param loss_value: 当前 mini-batch 的训练 loss 值
        :return:
        """
        info = [(f'{time.asctime(time.localtime(time.time()))} 第 {epoch} 次迭代的第 {idx} 个mini-batch的loss值为', loss_value)]
        file_path = os.path.join(self.result_dir_path, f'{self.__class__._mini_batch_loss_file_name}.csv')
        self.__save_csv(info, file_path)

    def _record_loss(self, epoch, loss):
        """
        记录一次迭代的Loss平均值
        :param epoch: 当前的迭代次数
        :param loss: 平均loss值
        :return:
        """
        info = [(f'{time.asctime(time.localtime(time.time()))} 第 {epoch} 次迭代的平均loss值为', loss)]
        file_path = os.path.join(self.result_dir_path, f'{self.__class__._epoch_loss_file_name}.csv')
        self.__save_csv(info, file_path)

    def __save_csv(self, info, file_path, mode='a'):
        """
        保存数据至csv文件
        :param info: 数据，格式为 [(d1_1,d1_2,...,d1_n),(d2_1,d2_2,...,d2_n),...]
        :param file_path: 保存的文件路径（包括文件名的绝对路径）
        :param mode: 写入的模式 w:覆盖写 a:追加写
        :return:
        """
        with open(file_path, mode=mode, newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerows(info)


if __name__ == '__main__':
   pass