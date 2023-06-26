import argparse
import numpy as np
import os
import torch
import torch.nn as nn
import random
from data.dataset import get_data
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from k_means_constrained import KMeansConstrained
from sklearn.metrics import accuracy_score
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

class k_means_fcnn(nn.Module):
    def __init__(self, in_channel, num_class):
        super(k_means_fcnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channel, 512, kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(512, 128, kernel_size=3, stride=1)
        self.conv3 = nn.Conv1d(128, 128, kernel_size=3, stride=1)
        self.fc = nn.Linear(128, num_class)
        self.avg_pool = nn.AvgPool1d(2, 2)

    def pad(self, x):
        size = x.shape[-1]
        out = torch.cat([x[:, :, size-1:], x, x[:,:, :1]], dim=-1)
        return out

    def forward(self, x):
        x = self.pad(x)
        x = self.conv1(x)
        x = self.avg_pool(x)
        x = self.pad(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        x = self.pad(x)
        x = self.conv3(x)
        x = self.avg_pool(x)
        x = x.squeeze(-1)
        x = self.fc(x)
        return x


class super_data(Dataset):
    def __init__(self, X, y, n_cluster=10):
        # self.cluster_sample = cluster(X, y, n_cluster)
        self.cluster_sample = no_cluster(X, y)
        self.label_unique = len(np.unique(y))
        self.size = 0

        for k, v in self.cluster_sample.items():
            self.size += v[0].shape[0] * 10
            # self.size += v.shape[0] // 10
        self.n_cluster = n_cluster
        # random_permute = np.random.permutation(self.label.size)
        # self.super_data = self.super_data[random_permute, :, :]
        # self.label = self.label[random_permute]

    def __getitem__(self, idx):
        choose_label = random.randint(0, self.label_unique - 1)
        choose_sample = self.cluster_sample[str(choose_label)]
        # cluster_size = choose_sample[0].shape[0]
        cluster_size = choose_sample.shape[0]
        random_choose = [random.randint(0, cluster_size - 1) for _ in range(self.n_cluster)]
        concat_data = []
        for i, (rand_idx) in enumerate(random_choose):
            # concat_data.append(choose_sample[i][rand_idx, :, :])
            concat_data.append(choose_sample[rand_idx, :])

        # concat_data = np.concatenate(concat_data, axis=1)
        concat_data = np.stack(concat_data, axis=-1)
        return (concat_data, choose_label)

    def __len__(self):
        return self.size



def cluster(X, y, n_cluster=10):
    idx_list, super_data, label = [], [], []
    for i in range(len(np.unique(y))):
        idx_list.append([index for (index, value) in enumerate(y) if value == i])

    cluster_sample = {}
    for ori_label, (index) in enumerate(idx_list):
        data = X[index, :]
        size = len(data) // n_cluster
        data = data[:size * n_cluster, :]
        clf = KMeansConstrained(n_clusters=n_cluster, size_max=size,  random_state=0)
        clf.fit_predict(data)
        sample, cat_tensor_i, result = [], [], []
        for i in range(n_cluster):
            idx = [index for (index, value) in enumerate(clf.labels_) if value == i]
            cat_tensor_i.append(data[idx, :, np.newaxis])
    #     random_choose = np.random.permutation(len(idx))
    #     for rand_idx in random_choose:
    #         for i in range(n_cluster):
    #             sample.append(cat_tensor_i[i][rand_idx, :, :])
    #         sample = np.stack(sample, axis=-1)
    #         result.append(sample.transpose(1,0,2))
    #         sample = []
    #     result = np.concatenate(result, axis=0)
    #     super_data.append(result)
    #     label.append([ori_label for _ in range(result.shape[0])])
    # return np.concatenate(super_data, axis=0), np.concatenate(label, axis=0)
        cluster_sample.update({str(ori_label): cat_tensor_i})
    return cluster_sample

def no_cluster(X, y):
    idx_list = []
    for i in range(len(np.unique(y))):
        idx_list.append([index for (index, value) in enumerate(y) if value == i])
    sample = {}
    for ori_label, (index) in enumerate(idx_list):
        data = X[index, :]
        sample.update({str(ori_label): data})

    return sample
if __name__ == '__main__':
    random.seed(5)

    parser = argparse.ArgumentParser(description='PyTorch v1.4, kmeans classfication Training')
    parser.add_argument('-d', '--dataset', type=str, required=False, default='cardio', help='Dataset Name')
    args = parser.parse_args()
    dataset_name = args.dataset
    regress = ['year', 'yahoo', 'MSLR', 'syn']
    print('===> Getting data ...')
    print('Dataset: ' + dataset_name)
    X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(dataset_name)
    train_set = super_data(X_train, y_train)
    test_set = super_data(X_test, y_test)
    train_loader = DataLoader(train_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False, num_workers=4, pin_memory=True)
    model = k_means_fcnn(in_channel=11, num_class=2).cuda()
    optimizer = Adam(model.parameters(), lr=0.0001)
    criterion = nn.CrossEntropyLoss()
    best = 0.0
    for epoch in range(0, 2000):
        loss_list = []
        model.train()
        for i, (input, label) in enumerate(train_loader):
            optimizer.zero_grad()
            input, label = input.cuda().float(), label.cuda().long()
            output = model(input)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.cpu().detach().item())

        print('epoch: ' + str(epoch))
        print('loss: ' + str(np.mean(loss_list)))
        model.eval()
        pred_list, label_list = [], []
        for i, (input, label) in enumerate(test_loader):
            input, label = input.cuda().float(), label.long()
            with torch.no_grad():
                output = model(input)
                pred = torch.max(output, 1)[1]
                pred_list.append(pred.cpu().detach().numpy())
                label_list.append(label.detach().numpy())
        y_true = np.hstack(label_list)
        y_score = np.hstack(pred_list)
        acc = accuracy_score(y_true, y_score)
        print('acc: ' + str(acc))
        if acc > best:
            best = acc
        print('Best acc: ' + str(best))

