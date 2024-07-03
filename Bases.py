import numpy as np
import torch
import yaml
from torch import optim, nn

import argparse
from FederatedTask import Cifar10FederatedTask, TinyImagenetFederatedTask, financeFederatedTask
from Params import Params
from Client import Client
from Server import ServerAvg

from metrics.accuracy_metric import AccuracyMetric
from metrics.metric import Metric
from metrics.test_loss_metric import TestLossMetric
from torch.nn import Module
from Attacks import Attacks
# Type Definition
# define all the operations on global models
from synthesizers.pattern_synthesizer import PatternSynthesizer
from torch.utils.data import Subset
from Reports import FLReport, save_report, load_report
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# random.seed(10)

def save_item(item, item_name):
    save_folder_name = './save_model'
    if not os.path.exists(save_folder_name):
        os.makedirs(save_folder_name)
    torch.save(item, os.path.join(save_folder_name, item_name + ".pt"))


class FederatedBackdoorExperiment:
    def __init__(self, params):
        # prepare for dataset and model
        self.params = params

        if params.task == 'CifarFed':
            self.task = Cifar10FederatedTask(params=params)
            print("Cifar!!!")
        elif params.task == 'ImageNetFed':
            self.task = TinyImagenetFederatedTask(params=params)
            print("ImageNet!!!")
        elif params.task == 'financeFed':
            self.task = financeFederatedTask(params=params)
            print("finance!!!")
        else:
            print("Not support dataset")
        print("Training Dataset:{}".format(params.task))
        self.task.init_federated_task()  #加载数据集数据；初始化类中的各个函数，如建立模型、建立优化器和损失函数等；设置输入维度input_shape

        base_model = self.task.build_model()  # 建模，对模型初始化
        base_optimizer = self.task.build_optimizer(base_model)  # 建立优化器
        splited_dataset = self.task.sample_dirichlet_train_data(params.n_clients)  # 为各个客户端分配数据
        server_sample_ids = splited_dataset[params.n_clients]  # 最后一个客户端即服务器，server_sample_ids即服务器所包含的样本列表

        for k, _v in splited_dataset.items():
            print(k, len(_v))  #打印客户端序号，及该客户端所包含的样本数，其中0-49为客户端，50为服务器

        print("build server:", len(server_sample_ids))  # 服务器包含的样本数
        server_dataset = None
        if not len(server_sample_ids) == 0:  # 如果不为零，创建一个子数据集server_dataset，从训练数据集中根据上述服务器所包含的样本索引进行选择
            server_dataset = Subset(self.task.train_dataset, server_sample_ids)  # 其中包含了服务器端所需的样本数据
        n_mal = params.n_malicious_client  # 恶意客户端数量，默认为4
        #print("server_dataset", server_dataset)
        self.server = ServerAvg(model=base_model, optimizer=base_optimizer, n_clients=params.n_clients,  # 对服务器初始化
                                chosen_rate=params.chosen_rate,
                                dataset=server_dataset, batch_size=params.batch_size, device=params.device)

        handcraft_trigger, distributed = self.params.handcraft_trigger, self.params.distributed_trigger  # 手动触发器，分布式触发器
        # print("handcraft_trigger:", handcraft_trigger, "distributed_trigger:", distributed)
        self.synthesizer = PatternSynthesizer(self.task, handcraft_trigger, distributed, (0, n_mal))  # 生成对抗性模式
        self.attacks = Attacks(params, self.synthesizer)  # 初始化

        self.clients = list()
        # 从全部客户端中选出规定数目的恶意客户端，将其序号存在变量中
        malicious_ids = np.random.choice(range(params.n_clients), params.n_malicious_client, replace=False)
        self.malicious_ids = malicious_ids
        i_mal = 0

        # 对于恶意客户端，设置后门；普通客户端正常初始
        for c in range(params.n_clients):  # 对于每个客户端
            sample_ids = splited_dataset[c]  # 当前客户端被分配的样本序号
            dataset = Subset(self.task.train_dataset, sample_ids)  # 从训练数据中挑出被选择的样本，存储在dataset中
            client_model = self.task.build_model()  # 客户端建模，对模型初始化
            client_optimizer = self.task.build_optimizer(client_model)  # 客户端优化器
            is_malicious = True if c in malicious_ids else False  # 判断当前客户端是否恶意，如果是，设置为true
            synthesizer, attacks = None, None
            if is_malicious:  # 如果是恶意客户端
                i_mal = i_mal + 1  # 该值为不为零时，下一行会多执行一个函数
                synthesizer = PatternSynthesizer(self.task, handcraft=handcraft_trigger, distributed=distributed,
                                                 mal=(i_mal, n_mal))

                print(synthesizer.get_pattern()[0].shape, synthesizer.get_pattern()[1].shape)
                attacks = Attacks(params, synthesizer)

            client = Client(model=client_model, client_id=c, optimizer=client_optimizer, is_malicious=is_malicious,
                            dataset=dataset, local_epoch=params.local_epoch, batch_size=params.batch_size,
                            attacks=attacks, device=params.device)  # 对客户端各项参数初始化

            self.clients.append(client)  # 将所有客户端加入列表
            print('build client:{}, mal:{}, data_num:{}\n'.format(c, is_malicious, len(dataset)))

    def fedavg_training(self, identifier=None):

        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):  # 依次遍历epoch
            print('Round {}: FedAvg Training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)  # 根据全局模型权重更新本地模型
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])  # 随机选择两个客户端
            for client in self.clients:  # 遍历所有客户端，即客户端训练阶段
                if client.client_id not in chosen_ids:  # 如果未被选中
                    client.idle()  # 记录轮数，更新学习率调度器
                else:
                    client.handcraft(self.task)  # 如果是良性客户端，什么也没做；如果是恶意客户端，向模型中加入后门，并计算神经元激活值
                    client.train(self.task,file_path)  # 对客户端进行训练，计算损失值
                # save_item(client.local_model, str(client.client_id))
                # print("saved")
            self.server.aggregate_global_model(self.clients, chosen_ids, None)  # 将被选中的客户端模型聚合到全局模型中
            print('Round {}: FedAvg Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False))
            fl_report.record_round_vars(self.test(epoch, backdoor=True))
            if (epoch + 1) % 50 == 0:  # 每50轮保存一次
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 30)

    def finetuning_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Finetuning Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})

            self.server.fine_tuning(self.task, self.clients, chosen_ids)
            print('Round {}: Finetuning Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 10 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def mitigation_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Mitigation Training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            # get the order
            prune_order = self.server.collect_conv_ranks(self.task, self.clients, chosen_ids, None)
            if epoch % 5 == 0:
                # try to prune
                self.server.conv_pruning(self.task, orders=prune_order)
                # adjust
                self.server.adjust_extreme_parameters(threshold=3)

            print('Round {}: Mitigation Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def mine_mitigation_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: mine training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            benign_ids = self.server.collect_benign_client(self.task, self.clients, chosen_ids, None)

            self.server.aggregate_global_model(self.clients, benign_ids, None)
            # get the order
            # prune_order = self.server.collect_conv_ranks(self.task, self.clients, chosen_ids, None)
            # if epoch % 1 == 0:
            #     # try to prune
            #     self.server.conv_pruning(self.task, orders=prune_order)
            #     # adjust
            #     self.server.adjust_extreme_parameters(threshold=3)

            print('Round {}: Mitigation Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 5 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 30)

    def mine2_mitigation_training(self, identifier):

        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):  # 依次遍历epoch
            print('Round {}: mine training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)  # 根据全局模型权重更新本地模型
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])  # 随机选择两个客户端
            for client in self.clients:  # 遍历所有客户端，即客户端训练阶段
                if client.client_id not in chosen_ids:  # 如果未被选中
                    client.idle()  # 记录轮数，更新学习率调度器
                else:
                    client.handcraft(self.task)  # 如果是良性客户端，什么也没做；如果是恶意客户端，向模型中加入后门，并计算神经元激活值
                    client.train(self.task,file_path)  # 对客户端进行训练，计算损失值

            print("chosen_ids:",chosen_ids)
            benign_ids = self.server.collect2_benign_client(self.task, self.clients, chosen_ids, None)

            self.server.aggregate_global_model(self.clients, benign_ids, None)  # 聚合良性客户端的模型
            # get the order
            # prune_order = self.server.collect_conv_ranks(self.task, self.clients, chosen_ids, None)
            # if epoch % 1 == 0:
            #     # try to prune
            #     self.server.conv_pruning(self.task, orders=prune_order)
            #     # adjust
            #     self.server.adjust_extreme_parameters(threshold=3)

            print('Round {}: mine2 Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})

            self.server.ensemble_distillation(self.task, self.clients, benign_ids)
            print('Round {}: Distillation Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('distill test:' + '\n')
                #.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})

            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 30)

    def ensemble_distillation_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Ensemble Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            self.server.ensemble_distillation(self.task, self.clients, chosen_ids)
            print('Round {}: Distillation Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 10 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def adaptive_distillation_training(self, identifier=None):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Adaptive Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)
            pts = self.server.get_median_scores(self.task, self.clients, chosen_ids)
            self.server.aggregate_global_model(self.clients, chosen_ids, pts)

            print('Round {}: FedAvg Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})

            self.server.adaptive_distillation(self.task, self.clients, chosen_ids)
            print('Round {}: FedRAD Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 10 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))
            print('-' * 50)

    def crfl_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: CRFL Distillation Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            self.server.aggregate_global_model(self.clients, chosen_ids, None)
            # print('Round {}: FedAvg Testing'.format(epoch))
            # fl_report.record_round_vars(self.crfl_test(epoch, backdoor=False), notation={'is_distill': False})
            # fl_report.record_round_vars(self.crfl_test(epoch, backdoor=True), notation={'is_distill': False})

            if not epoch == self.params.n_epochs - 1:
                self.server.clip_weight_norm()
                self.server.add_differential_privacy_noise(sigma=0.002, cp=False)
            print('Round {}: CRFL Testing'.format(epoch))
            fl_report.record_round_vars(self.crfl_test(epoch, backdoor=False), notation={'is_distill': True})
            fl_report.record_round_vars(self.crfl_test(epoch, backdoor=True), notation={'is_distill': True})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def deepsight_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Deep-Sight Training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            self.server.deepsight_aggregate_global_model(self.clients, chosen_ids, self.task, None)
            print('Round {}: Deep-Sight Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def robust_lr_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Robust LR Training'.format(epoch))

            with open(file_path, "a+") as f:
                f.write(str(epoch) + '\n')  # 将 fl_report 转换为字符串后写入文件
                f.write('train:' + '\n')
                # 不需要再次调用 f.close，使用 with open 语句会在结束时自动关闭文件

            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            flip_analysis = self.server.sign_voting_aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: Robust LR Testing'.format(epoch))

            with open(file_path, "a+") as f:
                f.write('test:' + '\n')
                #f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件

            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation=None)
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'flip_analysis': flip_analysis})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def bulyan_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Bulyan Training'.format(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task,file_path)

            self.server.bulyan_aggregate_global_model(self.clients, chosen_ids, None)
            print('Round {}: Bulyan Testing'.format(epoch))
            fl_report.record_round_vars(self.test(epoch, backdoor=False), notation={'is_distill': False})
            fl_report.record_round_vars(self.test(epoch, backdoor=True), notation={'is_distill': False})
            if (epoch + 1) % 50 == 0:
                saved_name = identifier + "_{}".format(epoch + 1)
                save_report(fl_report, './{}'.format(saved_name))

    def backdoor_unlearning_training(self, identifier):
        fl_report.create_record(identifier, checkout=True)
        fl_report.record_class_vars(self.params)

        for epoch in range(self.params.n_epochs):
            print('Round {}: Backdoor Unlearning Training'.foramt(epoch))
            self.server.broadcast_model_weights(self.clients)
            chosen_ids = self.server.select_participated_clients(fixed_mal=[])
            for client in self.clients:
                if client.client_id not in chosen_ids:
                    client.idle()
                else:
                    client.handcraft(self.task)
                    client.train(self.task)

            self.server.aggregate_global_model(self.clients, chosen_ids, None)

    def get_activation(self, backdoor, name):
        test_loader = self.task.test_loader
        local_model = torch.load('./save_model/' + name + '.pt')
        local_model.eval()
        final_activations = None
        for i, data in enumerate(test_loader):
            batch = self.task.get_batch(i, data)
            if backdoor:
                batch = self.attacks.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
            final_activation = local_model.final_activations(batch.inputs)
            # print(final_activation.shape)
            if final_activations is None:
                final_activations = torch.zeros_like(final_activation) + final_activation
            else:
                final_activations = final_activations + final_activation

            # if i + 1 == n_test_batch:
            #     break

        final_activations = final_activations / min(i, len(test_loader))
        channel_activations = torch.sum(final_activations, dim=[0, 2, 3])
        print(channel_activations)

    def test(self, epoch, backdoor, another_model=None):

        if self.params.handcraft and self.params.handcraft_trigger:
            self.attacks.synthesizer.pattern = 0
            for i in self.malicious_ids:  # 遍历所有恶意客户端
                self.attacks.synthesizer.pattern += self.clients[i].attacks.synthesizer.pattern  # 累加当前恶意客户端的攻击模式
            self.attacks.synthesizer.pattern = self.attacks.synthesizer.pattern / len(self.malicious_ids)  # 计算出平均模式

        target_model = self.server.global_model if another_model is None else another_model
        # target_model = torch.load('./save_model/6.pt')
        target_model.eval()  # 设置为评估状态
        test_loader = self.task.test_loader
        for metric in self.task.metrics:
            metric.reset_metric()  # 重置指标，便于收集新的指标数据
        with torch.no_grad():  # 确保在接下来的推理过程中不会进行梯度计算
            for i, data in enumerate(test_loader):  # 遍历测试数据批次
                batch = self.task.get_batch(i, data)
                # print(batch.labels)
                if backdoor:  # 如果 `backdoor` 为真，则使用攻击合成器创建一个带有后门的批处理数据
                    batch = self.attacks.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
                # print(batch.labels)
                outputs = target_model(batch.inputs)  # 对刚合成的该批数据进行模型推理得到输出
                '''To Modify'''
                self.task.accumulate_metrics(outputs=outputs, labels=batch.labels)  # 好像没有发挥作用
            print("backdoor:{} metric:{}".format(backdoor, self.task.metrics))

        round_info = dict()
        for metric in self.task.metrics:
            with open(file_path, "a+") as f:
                #f.write('test:' + '\n')
                f.write(str(metric) + '\n')  # 将 fl_report 转换为字符串后写入文件
            round_info.update(metric.get_value())
        round_info['backdoor'] = backdoor
        round_info['epoch'] = epoch
        return round_info

    def crfl_test(self, epoch, backdoor, another_model=None):
        if self.params.handcraft:
            self.attacks.synthesizer.pattern = 0
            for i in self.malicious_ids:
                self.attacks.synthesizer.pattern += self.clients[i].attacks.synthesizer.pattern
            self.attacks.synthesizer.pattern = self.attacks.synthesizer.pattern / len(self.malicious_ids)

        target_model = self.server.global_model if another_model is None else another_model
        target_model.eval()
        test_loader = self.task.test_loader

        smoothed_models = [self.server.add_differential_privacy_noise(sigma=0.002, cp=True) for m in range(5)]
        for metric in self.task.metrics:
            metric.reset_metric()
        with torch.no_grad():
            for i, data in enumerate(test_loader):
                batch = self.task.get_batch(i, data)
                if backdoor:
                    batch = self.attacks.synthesizer.make_backdoor_batch(batch, test=True, attack=True)
                outputs = 0
                for target_model in smoothed_models:
                    prob = torch.nn.functional.softmax(target_model(batch.inputs), 1)
                    # print("prob:",prob[0])
                    outputs = outputs + prob
                outputs = outputs / (len(smoothed_models))
                self.task.accumulate_metrics(outputs=outputs, labels=batch.labels)
            print("backdoor:{} metric:{}".format(backdoor, self.task.metrics))

        round_info = dict()
        for metric in self.task.metrics:
            round_info.update(metric.get_value())
        round_info['backdoor'] = backdoor
        round_info['epoch'] = epoch
        return round_info

    def make_criterion(self) -> Module:
        """Initialize with Cross Entropy by default.

        We use reduction `none` to support gradient shaping defense.
        :return:
        """
        return nn.CrossEntropyLoss(reduction='none')

    def generate_malicious_client_ids(self, n_client, n_malicious_client, test):
        if test:
            return [0]
        else:
            return np.random.choice(range(n_client), n_malicious_client, replace=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='pass in a parameter')
    parser.add_argument('--defence', type=str, help='defence name')
    parser.add_argument('--config', type=str, help='configs', choices=['cifar', 'imagenet', 'finance'])
    parser.add_argument('--backdoor', type=str, help='type of backdoor attacks',
                        choices=['neurotoxin', 'ff', 'dba', 'naive', 'baseline'])
    parser.add_argument('--model', type=str, help='model', choices=['simple', 'resnet18'])
    args = parser.parse_args()
    configs = 'configs/{}_fed.yaml'.format(args.config)
    with open(configs) as f:
        paramss = yaml.load(f, Loader=yaml.FullLoader)  # 将config文件加载到paramss中

    params = Params(**paramss)  # 在Params.py中对参数进行设置
    params.defence = args.defence
    params.model = args.model

    print(params.defence, args.defence)
    print("args server_dateset:{}".format(params.server_dataset))
    print("args model:{}".format(args.model))
    print("args batch_size:{}".format(params.batch_size))
    print("args defence:{}".format(params.defence))
    params.backdoor = args.backdoor
    if args.backdoor == 'ff':
        params.distributed_trigger = False  # 分布式触发器
        params.handcraft = True  # 手动设置
        params.handcraft_trigger = True  # 手动触发器
    elif args.backdoor == 'dba':
        params.distributed_trigger = True
        params.handcraft = False
        params.handcraft_trigger = False
    elif args.backdoor == 'naive':
        params.distributed_trigger = False
        params.handcraft = False
        params.handcraft_trigger = False
    elif args.backdoor == 'neurotoxin':
        params.distributed_trigger = False
        params.handcraft = False
        params.handcraft_trigger = False
    else:
        print("Not implemented defenses")

    fl_report = FLReport()  # ？
    experiment_name = "{}/{}_{}_{}_{}_h{}_c{}".format(params.resultdir, args.backdoor, args.defence, args.config,
                                                      args.model, params.heterogenuity, params.n_clients)  # 为实验命名
    experiment_name_1 = "{}_{}_{}_{}_h{}_c{}.txt".format(args.backdoor, args.defence, args.config,
                                                      args.model, params.heterogenuity, params.n_clients)  # 为实验命名, params.n_epochs
    file_path = os.path.join("", experiment_name_1)
    # print(experiment_name)
    experiment = FederatedBackdoorExperiment(params)

    # experiment.test(3, True)
    # experiment.test(3, False)
    # experiment.get_activation(True, '1')
    # experiment.get_activation(True, '4')
    # experiment.get_activation(True, '5')
    if params.defence == 'mine2':
        experiment.mine2_mitigation_training(identifier=experiment_name)
    elif params.defence == 'fedavg':
        experiment.fedavg_training(identifier=experiment_name)
    elif params.defence == 'ed':
        experiment.ensemble_distillation_training(identifier=experiment_name)
    elif params.defence == 'mediod-distillation':
        experiment.adaptive_distillation_training(identifier=experiment_name)
    elif params.defence == 'fine-tuning':
        experiment.finetuning_training(identifier=experiment_name)
    elif params.defence == 'mine_34':
        print('--------------------')
        experiment.mitigation_training(identifier=experiment_name)
    elif params.defence == 'mine':
        experiment.mine_mitigation_training(identifier=experiment_name)
    elif params.defence == 'robustlr':
        experiment.robust_lr_training(identifier=experiment_name)
    elif params.defence == 'certified-robustness':
        experiment.crfl_training(identifier=experiment_name)
    elif params.defence == 'bulyan':
        experiment.bulyan_training(identifier=experiment_name)
    elif params.defence == 'deep-sight':
        experiment.deepsight_training(identifier=experiment_name)
        print("Defence Name Errors")
