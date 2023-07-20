
import torch
import os
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader


class PCGDataset(Dataset):
    def __init__(self, root, fold_path, train):
        super(PCGDataset, self).__init__()

        self.train = train
        self.fold_path = fold_path
        self.root = root
        if self.train:
            self.csv_path = os.path.join(fold_path, 'train.csv')  # csv_path就是这一折里面的csv的路径
        else:
            self.csv_path = os.path.join(fold_path, 'val.csv')

        self.all_data_path, self.all_label, self.all_file_name = self.load_csv(self.csv_path)  # 这里就是得到两个list，里面放的是数据的路径和标签

    def load_csv(self, path):
        data_path = []
        label = []
        file_name = []
        with open(path, 'r') as f:
            data = f.read()
        for i in data.split('\n'):         # 这里注意要判空

            if len(i) == 0:
                continue
            else:
                data_path.append(i.split(',')[0])

                label.append(i.split(',')[1])
                temp = i.split('.')[0]

                file_name.append(temp.split('/')[1])

        return data_path, label, file_name

    def __len__(self):
        return len(self.all_data_path)

    def __getitem__(self, index):
        data_path = self.all_data_path[index]
        label = self.all_label[index]
        absolute_path = os.path.join(self.root, data_path)  # 这是数据的绝对路径
        data = np.load(absolute_path, allow_pickle=True)
        data = torch.from_numpy(data.astype(np.float32))
        # data = data.unsqueeze(0)   # data=>[1, 6000]
        # data = data.unsqueeze(1)   # data=>[6000, 1]

        return data, torch.tensor(int(label), dtype=torch.long), self.all_file_name[index]


if __name__ == '__main__':
    base_path = r'D:\Pcg_code\PCG_DATA_CUT'
    base_path2 = r'D:\Pcg_code\PCG_MFCC_DATA'

    model = PCGDataset(root=base_path2, fold_path=os.path.join(base_path2, 'fold_dataset_0'), train=True)  # 这里就是实例化dataset
    loader = DataLoader(model, batch_size=32, shuffle=True, num_workers=2)
    for data, label, name in loader:
        print(data.shape)   # [b, 39, 299]  这里是由299是由于我们有一个窗口，所以会从6000减少到299


