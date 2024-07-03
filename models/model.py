import torch.nn as nn


def is_bn_relavent(layer):
    if 'running'  in layer or 'tracked' in layer:
        return True
    else:
        return False


class Model(nn.Module):
    """
    Base class for models with added support for GradCam activation map
    and a SentiNet defense. The GradCam design is taken from:
https://medium.com/@stepanulyanin/implementing-grad-cam-in-pytorch-ea0937c31e82
    If you are not planning to utilize SentiNet defense just import any model
    you like for your tasks.
    """

    def __init__(self):
        super().__init__()
        self.gradient = None

    def activations_hook(self, grad):  # 获取激活值的梯度
        self.gradient = grad

    def get_gradient(self):  # 返回梯度值
        return self.gradient

    def get_activations(self, x):  # 获取激活值
        return self.features(x)

    def switch_grads(self, enable=True):  # 用于控制梯度的开关
        for i, p in self.named_parameters():
            p.requires_grad_(enable)

    def features(self, x):  # 获取特征
        """
        Get latent representation, eg logit layer.
        :param x:
        :return:
        """
        raise NotImplemented

    def final_activations(self, x):  # 获取最终的激活值
        raise NotImplemented

    def first_activations(self, x):  # 获取最初的激活值
        raise NotImplemented

    def forward(self, x, latent=False):  # 前向传播
        raise NotImplemented
