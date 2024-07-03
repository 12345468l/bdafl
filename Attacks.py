import copy
import random
from collections import defaultdict, OrderedDict
from typing import Dict

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from FederatedTask import Cifar10FederatedTask, TinyImagenetFederatedTask, financeFederatedTask
from models.extractor import FeatureExtractor
from models.model import Model
from models.nc_model import NCModel
from Params import Params
from models.resnet import layer2module, ResNet, resnet18
from models.simple import SimpleNet
from synthesizers.synthesizer import Synthesizer
import numpy as np
from losses.loss_functions import trigger_attention_loss, trigger_loss


# from scipy import stats

def get_accuracy(model, task, loader):
    for metric in task.metrics:
        metric.reset_metric()

    model.eval()
    specified_metrics = ['AccuracyMetric']
    for i, data in enumerate(loader):
        batch = task.get_batch(i, data)
        outputs = model(batch.inputs)
        '''To Modify'''
        task.accumulate_metrics(outputs, batch.labels, specified_metrics=specified_metrics)

    accuracy = None
    for metric in task.metrics:
        if metric.__class__.__name__ in specified_metrics:
            accuracy = metric.get_value()

    return accuracy['Top-1']


def test_handcrafted_acc(model, target, id, task, loader):
    weights = model.state_dict()
    cur_conv_kernel = weights[target][id, ...].clone().detach()
    weights[target][id, ...] = 0
    accuracy = get_accuracy(model, task, loader)
    weights[target][id, ...] = cur_conv_kernel
    return accuracy


def get_conv_weight_names(model: Model):
    conv_targets = list()
    weights = model.state_dict()  # 获取模型参数
    for k in weights.keys():  # 识别所有卷积层的权重参数
        if 'conv' in k and 'weight' in k:
            conv_targets.append(k)

    return conv_targets


def get_neuron_weight_names(model: Model):  # 返回模型全连接层参数权重
    neuron_targets = list()
    weights = model.state_dict()
    for k in weights.keys():
        if 'fc' in k and 'weight' in k:
            neuron_targets.append(k)

    return neuron_targets


class Attacks:
    params: Params
    synthesizer: Synthesizer
    nc_model: Model
    nc_optimzer: torch.optim.Optimizer
    nc_p_norm: int
    acc_threshold: int

    def __init__(self, params, synthesizer):
        self.params = params
        self.synthesizer = synthesizer
        self.loss_tasks = self.params.loss_tasks.copy()
        self.loss_balance = self.params.loss_balance
        self.mgda_normalize = self.params.mgda_normalize
        self.backdoor_label = params.backdoor_label
        self.handcraft = params.handcraft
        self.acc_threshold = params.acc_threshold if params.handcraft else 0
        self.handcraft_trigger = params.handcraft_trigger
        self.kernel_selection = params.kernel_selection
        self.raw_model = None
        self.neurotoxin = True if params.backdoor == 'neurotoxin' else False

        if 'neural_cleanse' in self.params.loss_tasks:
            self.nc_model = NCModel(params.input_shape[1]).to(params.device)
            self.nc_optim = torch.optim.Adam(self.nc_model.parameters(), 0.01)
        if 'mask_norm' in self.params.loss_tasks:
            self.nc_p_norm = self.params.nc_p_norm
        if self.kernel_selection == "movement":
            self.previous_global_model = None

    def scale_losses(self, loss_tasks, loss_values, scale):
        blind_loss = 0
        for it, t in enumerate(loss_tasks):
            self.params.running_losses[t].append(loss_values[t].item())  # 损失值
            self.params.running_scales[t].append(scale[t])  # 比例尺度

            if it == 0:
                blind_loss = scale[t] * loss_values[t]
            else:
                blind_loss += scale[t] * loss_values[t]

        self.params.running_losses['total'].append(blind_loss.item())
        return blind_loss

    def search_candidate_weights(self, model: Model, proportion=0.2):
        assert self.kernel_selection in ['random', 'movement']
        candidate_weights = OrderedDict()  # 创建了一个空的有序字典
        model_weights = model.state_dict()  # 获取客户端本地模型权重

        n_labels = 0

        if self.kernel_selection == "movement":
            # 第一个epoch时，由于previous_global_model为空，在client.py中被设置成本地模型的克隆版本
            history_weights = self.previous_global_model.state_dict()
            for layer in history_weights.keys():  # 遍历模型权重的每一层
                if 'conv' in layer:  # 根据层的类型设置了不同的比例
                    proportion = self.params.conv_rate
                elif 'fc' in layer:
                    proportion = self.params.fc_rate

                candidate_weights[layer] = (model_weights[layer] - history_weights[layer]) * model_weights[layer]
                n_weight = candidate_weights[layer].numel()  # 计算候选权重中的元素数目
                # 首先通过flatten()将候选权重展平为一维张量，然后使用torch.sort对其进行升序排序。接着，取排序后的张量中的第n_weight * proportion个元素作为阈值theta
                theta = torch.sort(candidate_weights[layer].flatten(), descending=False)[0][int(n_weight * proportion)]
                candidate_weights[layer][candidate_weights[layer] < theta] = 1  # 小于阈值的权重设置为1，其余设置为0
                candidate_weights[layer][candidate_weights[layer] != 1] = 0

        return candidate_weights

    def flip_filter_as_trigger(self, conv_kernel: torch.Tensor, difference):
        flip_factor = self.params.flip_factor
        c_min, c_max = conv_kernel.min(), conv_kernel.max()
        pattern = None
        if difference is None:
            pattern_layers, _ = self.synthesizer.get_pattern()
            x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top
            x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot
            pattern = pattern_layers[:, x_top:x_bot, y_top:y_bot]
        else:
            pattern = difference
        w = conv_kernel[0, ...].size()[0]
        resize = transforms.Resize((w, w))
        pattern = resize(pattern)
        p_min, p_max = pattern.min(), pattern.max()
        scaled_pattern = (pattern - p_min) / (p_max - p_min) * (c_max - c_min) + c_min

        crop_mask = torch.sign(scaled_pattern) != torch.sign(conv_kernel)
        conv_kernel = torch.sign(scaled_pattern) * torch.abs(conv_kernel)
        conv_kernel[crop_mask] = conv_kernel[crop_mask] * flip_factor
        return conv_kernel

    def calculate_activation_difference(self, raw_model, new_model, layer_name, kernel_ids, task, loader: DataLoader):
        raw_extractor, new_extractor = FeatureExtractor(raw_model), FeatureExtractor(new_model)
        raw_extractor.insert_activation_hook(raw_model)
        new_extractor.insert_activation_hook(new_model)
        difference = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            raw_outputs = raw_model(batch.inputs)
            new_outputs = new_model(batch.inputs)
            module = layer2module(new_model, layer_name)
            # modify this
            raw_batch_activations = raw_extractor.activations(raw_model, module)[:, kernel_ids, ...]
            new_batch_activations = new_extractor.activations(new_model, module)[:, kernel_ids, ...]
            batch_activation_difference = new_batch_activations - raw_batch_activations
            # mean_difference = torch.mean(batch_activation_difference, [0, 1])
            mean_difference = torch.mean(batch_activation_difference, [0])
            difference = difference + mean_difference if difference is not None else mean_difference

        difference = difference / len(loader)

        raw_extractor.release_hooks()
        new_extractor.release_hooks()

        return difference

    def conv_features(self, model, task, loader, attack):  # 计算卷积特征的平均值
        features = None
        avg_features = None
        if isinstance(model, SimpleNet):
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                feature = model.features(batch.inputs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)
        if isinstance(model, ResNet):
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
                feature = model.features(batch.inputs).mean([0])
                features = feature if features is None else features + feature
            avg_features = features / len(loader)

        return avg_features

    def calculate_feature_difference(self, raw_model, new_model, task, loader):
        diffs = None
        if isinstance(raw_model, SimpleNet) and isinstance(new_model, SimpleNet):
            for i, data in enumerate(loader):
                batch = task.get_batch(i, data)
                batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
                diff = new_model.features(batch.inputs).mean([0]) - raw_model.features(batch.inputs).mean([0])
                diffs = diff if diffs is None else diffs + diff
        elif isinstance(raw_model, ResNet) and isinstance(new_model, ResNet):
            raise NotImplemented
        avg_diff = diffs / len(loader)
        return avg_diff

    def conv_activation(self, model, layer_name, task, loader, attack):  # 获取给定模型在特定层的平均激活值
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)
        conv_activations = None
        #print("run1")
        #print("0",loader)
        for i, data in enumerate(loader):
            #print("run")
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)  # 插入了后门的批次
            _ = model(batch.inputs)
            conv_activation = extractor.activations(model, module)
            conv_activation = torch.mean(conv_activation, [0])
            #print("end")
            #print("111", conv_activation)
            conv_activations = conv_activation if conv_activations is None else conv_activations + conv_activation

        #print("1111",conv_activations)
        #print("222",loader)
        #print("end1")
        avg_activation = conv_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def fc_activation(self, model: Model, layer_name, task, loader, attack):
        extractor = FeatureExtractor(model)
        hook = extractor.insert_activation_hook(model)
        module = layer2module(model, layer_name)

        print("module:",module)

        neuron_activations = None
        for i, data in enumerate(loader):
            batch = task.get_batch(i, data)
            batch = self.synthesizer.make_backdoor_batch(batch, test=True, attack=attack)
            _ = model(batch.inputs)
            neuron_activation = extractor.activations(model, module)
            neuron_activations = neuron_activation if neuron_activations is None else neuron_activations + neuron_activation

        avg_activation = neuron_activations / len(loader)
        extractor.release_hooks()
        torch.cuda.empty_cache()
        return avg_activation

    def inject_handcrafted_filters(self, model, candidate_weights, task, loader):  # model为本地模型
        conv_weight_names = get_conv_weight_names(model)  # 包含了卷积层的权重参数
        difference = None
        for layer_name, conv_weights in candidate_weights.items():  # 遍历每个卷积层
            if layer_name not in conv_weight_names:  # 如果该层不在conv_weight_names，就跳过本次循环
                continue
            model_weights = model.state_dict()  # 获取模型参数
            n_filter = conv_weights.size()[0]
            for i in range(n_filter):  # 对于当前卷积层的每个滤波器
                conv_kernel = model_weights[layer_name][i, ...].clone().detach()  # 获取当前滤波器的权重
                # flip filter 翻转滤波器
                handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference)
                # handcrafted_conv_kernel = conv_kernel

                mask = conv_weights[i, ...]
                model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * \
                                                    model_weights[layer_name][i, ...]  # 生成新的滤波器
                # model_weights[layer_name][i, ...].mul_(1-mask)
                # model_weights[layer_name][i, ...].add_(mask * handcrafted_conv_kernel)

            model.load_state_dict(model_weights)  # 获取新的模型参数
            # 计算经过修改后的模型在给定任务和数据集上是否插入后门之间的激活值的差异
            #print("t",loader)
            difference = (self.conv_activation(model, layer_name, task, loader, True) -
                          self.conv_activation(model, layer_name, task, loader, False))

            print("handcraft_conv: {}".format(layer_name))

        torch.cuda.empty_cache()
        if difference is not None:  # 是否插入后门之间的卷积特征平均值的差异
            feature_difference = self.conv_features(model, task, loader, True) - self.conv_features(model, task, loader,
                                                                                                    False)
            return feature_difference

    def set_handcrafted_filters2(self, model: Model, candidate_weights, layer_name):
        conv_weights = candidate_weights[layer_name]  # 卷积层1按阈值划分后的模型权重（值为1或0）
        # print("check candidate:",int(torch.sum(conv_weights)))
        model_weights = model.state_dict()  # 本地模型权重
        temp_weights = copy.deepcopy(model_weights[layer_name])

        n_filter = conv_weights.size()[0]  # 获取了卷积层权重的深度，即滤波器数量

        for i in range(n_filter):  # 遍历每个滤波器
            conv_kernel = model_weights[layer_name][i, ...].clone().detach()  # 克隆并分离模型卷积层1当前滤波器的权重
            handcrafted_conv_kernel = self.flip_filter_as_trigger(conv_kernel, difference=None)  # 创建了一个经过处理的卷积核
            mask = conv_weights[i, ...]  # 当前滤波器按阈值划分后的权重
            model_weights[layer_name][i, ...] = mask * handcrafted_conv_kernel + (1 - mask) * model_weights[layer_name][
                i, ...]  # 本地模型中卷积层1当前滤波器的新权重

        # 更新模型的权重和偏置，使其与model_weights中存储的数值相匹配
        model.load_state_dict(model_weights)
        # n_turn=int(torch.sum(torch.sign(temp_weights)!=torch.sign(model_weights[layer_name])))
        # print("check modify:",n_turn)

    def optimize_backdoor_trigger(self, model: Model, candidate_weights, task, loader):
        pattern, mask = self.synthesizer.get_pattern()  # 如果后门位置不是动态的，不执行任何操作，直接返回self.synthesizer.pattern/mask值
        pattern.requires_grad = True  # 这意味着在反向传播过程中，`pattern` 将会计算梯度

        x_top, y_top = self.synthesizer.x_top, self.synthesizer.y_top  # 放入后门的起始位置
        x_bot, y_bot = self.synthesizer.x_bot, self.synthesizer.y_bot  # 放入后门的结束位置

        cbots, ctops = list(), list()
        for h in range(pattern.size()[0]):  # 遍历 `pattern` 的第一个维度
            cbot = (0 - task.means[h]) / task.lvars[h]
            ctop = (1 - task.means[h]) / task.lvars[h]
            cbots.append(round(cbot, 2))
            ctops.append(round(ctop, 2))

        raw_weights = copy.deepcopy(model.state_dict())  # 本地模型的权重
        self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")  # 对卷积层1的权重更新
        for epoch in range(2):  # 包含两个epoch的训练
            losses = list()
            for i, data in enumerate(loader):  # handcraft_loader测试数据集加载器，遍历每个数据批次
                batch_size = self.params.batch_size

                # 获取干净数据和带后门的数据，get_batch功能是将数据转移到cuda上
                # 但是这里都是从同一数据源data获取，实际并未达到这一效果？
                # 其实是准备两份相同的数据，一份不做任何处理：干净数据，一份做处理：带后门的数据
                clean_batch, backdoor_batch = task.get_batch(i, data), task.get_batch(i, data)

                # 分别对数据的输入值和标签值做后门处理
                # 对于前 `batch_size` 个输入，应用掩码和模式做变换得到新的输入值
                backdoor_batch.inputs[:batch_size] = (1 - mask) * backdoor_batch.inputs[:batch_size] + mask * pattern
                # 将后门批次数据标签的前 `batch_size` 个标签设置为指定的后门标签值
                backdoor_batch.labels[:batch_size].fill_(self.params.backdoor_label)
                # print(backdoor_batch.labels[:batch_size])

                self.set_handcrafted_filters2(model, candidate_weights, "conv1.weight")  # 对卷积层1的权重更新

                # loss, grads = trigger_attention_loss(raw_model, model, backdoor_batch.inputs, pattern, grads=True)
                # 计算了后门输入和干净输入的激活之间的差异的平方和，作为损失值
                loss, grads = trigger_loss(model, backdoor_batch.inputs, clean_batch.inputs, pattern, grads=True)
                losses.append(loss.item())

                pattern = pattern + grads[0] * 0.1  # 对模式更新

                n_channel = pattern.size()[0]  # 模式的通道数
                for h in range(n_channel):  # 对于每个通道h
                    # 在特定的区域(x_top:x_bot, y_top:y_bot)内，使用torch.clamp函数将模式的值限制在一定范围内，范围由cbots[h]和ctops[h]决定
                    pattern[h, x_top:x_bot, y_top:y_bot] = torch.clamp(pattern[h, x_top:x_bot, y_top:y_bot], cbots[h],
                                                                       ctops[h], out=None)

                model.zero_grad()
            print("epoch:{} trigger loss:{}".format(epoch, np.mean(losses)))

        print(pattern[0, x_top:x_bot, y_top:y_bot].cpu().data)

        self.synthesizer.pattern = pattern.clone().detach()  # 将模式的克隆版本(不带梯度信息)赋值给前者
        self.synthesizer.pattern_tensor = pattern[x_top:x_bot, y_top:y_bot]

        model.load_state_dict(raw_weights)  # 更新模型的权重和偏置，使其与raw_weights中存储的数值相匹配
        torch.cuda.empty_cache()  # 释放未使用内存

    def inject_handcrafted_neurons(self, model, candidate_weights, task, diff, loader):
        handcrafted_connectvites = defaultdict(list)
        target_label = self.params.backdoor_label
        n_labels = -1
        if isinstance(task, Cifar10FederatedTask):
            n_labels = 10
        elif isinstance(task, TinyImagenetFederatedTask):
            n_labels = 200
        elif isinstance(task, financeFederatedTask):
            n_labels = 2

        fc_names = get_neuron_weight_names(model)  # 模型全连接层权重
        fc_diff = diff
        last_layer, last_ids = None, list()
        for layer_name, connectives in candidate_weights.items():
            if layer_name not in fc_names:  # 如果不是全连接层
                continue
            raw_model = copy.deepcopy(model)
            model_weights = model.state_dict()
            ideal_signs = torch.sign(fc_diff)   # 如果输入为正，返回1；如果输入为负，返回-1；如果输入为零，返回0
            n_next_neurons = connectives.size()[0]  # 代表connectives中神经元的数量
            # last_layer
            if n_next_neurons == n_labels:
                break

            print(n_next_neurons)
            print("------connectives:", connectives.size())
            #print("----------", ideal_signs.size())
            ideal_signs_1 = ideal_signs.repeat(n_next_neurons, 1)
            print("--------------------ideal_signs_1:", ideal_signs_1.size())
            #ideal_signs_1 = ideal_signs_1.reshape(2, 512, -1).mean(dim=2)

            ideal_signs = ideal_signs_1 * connectives
            #ideal_signs = ideal_signs.repeat(n_next_neurons, 1) * connectives
            # count the flip num 确定需要翻转的神经元的数量
            print("layer_name:",layer_name)
            n_flip = torch.sum(((ideal_signs * torch.sign(model_weights[layer_name]) * connectives == -1).int()))
            print("n_flip in {}:{}".format(layer_name, n_flip))
            model_weights[layer_name] = (1 - connectives) * model_weights[layer_name] + torch.abs(
                connectives * model_weights[layer_name]) * ideal_signs  # 计算新的模型权重
            model.load_state_dict(model_weights)  # 更新模型
            last_layer = layer_name
            fc_diff = self.fc_activation(model, layer_name, task, loader, attack=True).mean([0]) - self.fc_activation(
                model, layer_name, task, loader, attack=False).mean([0])  # 计算全连接层神经元平均激活值

    def fl_scale_update(self, local_update: Dict[str, torch.Tensor]):
        for name, value in local_update.items():
            value.mul_(self.params.fl_weight_scale)
