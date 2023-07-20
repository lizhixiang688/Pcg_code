# -*- coding: UTF-8 -*-
"""
@version: 1.0
@PackageName: hs_model - val.py
@author: yonghao
@Description: 验证模型性能的类
@since 2021/09/14 17:03
"""
import torch
import csv
import os
from sklearn.metrics import confusion_matrix
import pandas as pd


class Val:

    def __init__(self, dataloader, device, result_dir_path):
        self.is_cut = True
        self.device = device
        self.dataloader = dataloader
        self.result_dir_path = result_dir_path

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    # @torch.no_grad()  # 该装饰器可关闭梯度计算
    def run(self, model, index_file_name):
        """
        使用数据集验证模型
        :param model: model对象
        :param index_file_name: 指标保存的文件名
        :return:
        acc, ppv, se, spe, f1_score
        """
        # 关闭训练模式，禁止 batch-norm、dropout
        model.train(mode=False)
        logits = []
        gt = []
        f_n = []

        with torch.no_grad():
            # mini-batch训练
            for idx, (data, labels, filenames) in enumerate(self.dataloader):
                # step1:将 数据+标签 加入gpu进行运算
                data, labels = data.to(self.device), labels.to(self.device)

                # step2: 模型运行生成逻辑值等数据
                out = model(data)

                if isinstance(out, (list, tuple)):
                    out = out[0]  # 选取数据

                logits.append(out)
                gt.append(labels)
                f_n.extend(filenames)

            # 合并所有数据的信息 logits [total_data_N,classes_nums]  gt [total_data_N]
            logits = torch.cat(logits, dim=0)
            gt = torch.cat(gt, dim=0)
            # f_n = [n.split('_')[0] for n in f_n]  # 仅获取原始数据名，去除切片索引号
            if self.is_cut:
                pred, gt, f_n = self.__statistics(logits, gt, f_n)
            else:
                pred = torch.softmax(logits, dim=1).argmax(dim=1)  # 计算预测值

            # 获得评估指标
            acc, ppv, se, spe, f1_score, TP, TN, FP, FN, error_f_n = self._evaluate(pred, gt, f_n)

            # 保存指标数据
            self._save_index(acc, ppv, se, spe, f1_score, TP, TN, FP, FN, index_file_name=index_file_name)

            # 保存分错的数据信息
            with open(os.path.join(self.result_dir_path, 'error_result_info.csv'), mode='a', encoding='utf8',
                      newline='') as f:
                writer = csv.writer(f)
                writer.writerow(error_f_n)

            return acc, ppv, se, spe, f1_score

    def __statistics(self, logits, gt, filenames) -> tuple:
        """
        分片情况下的统计评估
        :param logits: 模型输出的总逻辑值 [data_N,class_nums]
        :param gt: 标签值 [data_N]
        :param filenames: [data_N]
        :return:
        """

        def vote(arr):
            arr_v = arr.map(lambda x: float(x.split(',')[0]))
            arr_l = arr.map(lambda x: int(x.split(',')[1]))
            ab_n = arr_l[arr_l == 1].count()
            n = arr_l[arr_l == 0].count()
            if ab_n > n:
                return 1
            elif ab_n < n:
                return 0

            # 平票的情况
            # 统计异常预测平均值
            ab_n_mean_pred = arr_v[arr_l == 1].mean()
            # 统计正常预测的平均值
            n_mean_pred = arr_v[arr_l == 0].mean()
            if ab_n_mean_pred >= n_mean_pred:
                return 1
            else:
                return 0

        # assert logits.size[0] == len(filenames) and len(filenames) == len(gt)
        pred = torch.softmax(logits, dim=1).max(dim=1)
        pred_v = pred.values.data.cpu().numpy()
        pred_l = pred.indices.data.cpu().numpy()
        pred_ = [f'{i},{pred_l[idx]}' for idx, i in enumerate(pred_v)]
        pd_result = pd.DataFrame({'filename': filenames, 'pred': pred_, 'label': gt.data.cpu().numpy()})
        group_by_filename = pd_result.groupby(pd_result['filename'].map(lambda x: x.split('_')[0]))
        agg_result = group_by_filename['pred'].agg(vote)
        agg_label = group_by_filename['label'].agg(lambda s: s.iloc[0])
        agg_filename = group_by_filename['filename'].agg(lambda s: s.iloc[0])

        return torch.tensor(agg_result, dtype=torch.float64).to(logits.device), \
               torch.tensor(agg_label, dtype=torch.long).to(logits.device), \
               [f.split('_')[0] for f in list(agg_filename)]

    def _save_index(self, *args, index_file_name):
        """
        保存指标数据
        :param args: acc, ppv, se, spe, f1_score, TP, TN, FP, FN
        :param index_file_name: 指标保存的文件名
        :return:
        """
        info = [args]
        path = os.path.join(self.result_dir_path, f'{index_file_name}.csv')
        self.__save_csv(info, path)

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

    def _calculate_metric(self, pred, label):
        """
        计算混淆矩阵

        :param pred: 预测值
        :param label: 标签值
        :return:
        """
        # tn, fp, fn, tp
        # confusion = confusion_matrix(label, pred, labels=(0, 1))
        TN, FP, FN, TP = confusion_matrix(label, pred, labels=(0, 1)).ravel()
        # TN = confusion[0, 0]
        # FP = confusion[0, 1]
        # FN = confusion[1, 0]
        # TP = confusion[1, 1]
        return TP, TN, FP, FN

    def _evaluate(self, pred, label, f_n):
        """
        评估方法
        :param pred: [batch,2] 模型输出
        :param label: [batch] 数据标签
        :param f_n: 数据的名称list
        :return:
        1. 准确率 Acc
            (TP + TN) / (TP + FP + TN + FN)

        2. 精确率 PPv（精确率指模型预测为正的样本中实际也为正的样本占被预测为正的样本的比例）
            TP / (TP + FP)

        3. 召回率(灵敏度 Se)（召回率指实际为正的样本中被预测为正的样本所占实际为正的样本的比例）
            TP / (TP + FN) 也称 敏感度(Se)

        4. 特异性(Spe)
            TN / (TN + FP)

        5. f1_score（F1 score是精确率和召回率的调和平均值）
            2 * PPv * Se/(PPv + Se)
        """

        # 获取分错类别的数据名称
        result = torch.eq(pred, label)
        error_f_n = [f_n[idx] for idx, r in enumerate(result) if not r]

        TP, TN, FP, FN = self._calculate_metric(pred.data.cpu().numpy(), label.data.cpu().numpy())

        acc = (TP + TN) / (TP + FP + TN + FN)
        ppv = TP / (TP + FP)
        se = TP / (TP + FN)
        spe = TN / (TN + FP)
        f1_score = 2 * ppv * se / (ppv + se)
        return acc, ppv, se, spe, f1_score, TP, TN, FP, FN, error_f_n


if __name__ == "__main__":
    pass


