import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model import Model


class SimpleNet(Model):
    def __init__(self, num_classes):
        super().__init__()
        # self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        torch.cuda.empty_cache()
        if num_classes == 10:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.fc1 = nn.Linear(8 * 8 * 64, 512)
            self.fc2 = nn.Linear(512, num_classes)
        elif num_classes == 200:
            self.conv1 = nn.Conv2d(3, 24, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(24, 50, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.fc1 = nn.Linear(16 * 16 * 50, 1024)
            self.fc2 = nn.Linear(1024, num_classes)
        elif num_classes == 2:
            self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=1)
            self.fc1 = nn.Linear(32 * 75 * 125, 512)
            self.fc2 = nn.Linear(512, num_classes)

        # self.fc3 = nn.Linear(32, num_classes)

    def first_activations(self, x):
        x = F.relu(self.conv1(x))
        return x

    def final_activations(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        return x

    # return the activations after the last conv_layers
    def features(self, x, layer=None):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(x.size()[0], -1)

        return x

    def forward(self, x, latent=False):
        x = F.relu(self.conv1(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        # print(x.shape)
        x = F.max_pool2d(x, 2, 2)
        # print(x.shape)
        if x.requires_grad:
            x.register_hook(self.activations_hook)

        x = x.view(x.size()[0], -1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        # print(x.shape)
        out = self.fc2(x)
        # print(x.shape)
        # out = F.log_softmax(x, dim=1)
        # print("outsize:",out.size())
        if latent:
            return out, x
        else:
            return out


def get_filter_ranks(model):
    model_weight = model.state_dict()
    filter_weight1 = model_weight['conv2.weight']
    filter_weight = torch.sum(filter_weight1, dim=[1, 2, 3])
    return filter_weight


def distance_ranks(stacked_vectors):
    # 计算这十个向量的平均值
    mean_vector = torch.mean(stacked_vectors, dim=0)

    # 计算每个向量与平均值的欧氏距离
    distances = [torch.dist(vec, mean_vector, 2) for vec in stacked_vectors]
    return distances


def get_benign_ids(list):
    from collections import Counter
    counter = Counter(list)

    # 找到出现次数最少的数字的出现次数
    min_count = min(counter.values())

    # 找到出现次数最少的数字
    min_count_numbers = [num for num, count in counter.items() if count == min_count]

    return min_count_numbers


if __name__ == '__main__':
    a = get_benign_ids([1,1,1,2,3])
    print(a)

    # final_activations = np.arange(48).reshape(2, 2, 3, 4)
    # print(final_activations)
    # final_activations = torch.tensor(final_activations)
    # final_activations = torch.sum(final_activations, dim=[2, 3])
    # print(final_activations)
    # final_activations = torch.sum(final_activations, dim=[0])
    # print(final_activations)
    # final_activations = torch.sum(final_activations, dim=[1])
    # print(final_activations)
    net = SimpleNet(10)
    # i = get_filter_ranks(net)
    # print(i.shape)
    # x = torch.randn(1, 3, 32, 32)
    model_weights = net.state_dict()
    for k, v in model_weights.items():
        print(k, v.shape)
    # a = model_weights['conv2.weight'][0]
    # print(a.shape, model_weights['conv2.weight'].shape)

    # x = net(x)
    # print("1",x.shape)
    # x = F.max_pool2d(x, 2, 2)
    # print(x.shape)
    #
    # target = net.state_dict()
    # for k, v in target.items():
    #     print(k, type(v), v.shape)

    # final_activations = net.final_activations(x)
    # print(final_activations.shape)
    # channel_activations = torch.sum(final_activations, dim=[0, 2, 3])
    # _, channel_idxs = torch.sort(channel_activations, descending=False)
    # _, channel_ranks = torch.sort(channel_idxs)
    # print(channel_ranks, channel_ranks.__len__() )
