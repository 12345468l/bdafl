import random

import torch
from torchvision.transforms import transforms, functional

from synthesizers.synthesizer import Synthesizer
from FederatedTask import Cifar10FederatedTask, TinyImagenetFederatedTask, financeFederatedTask
from FederatedTask import FederatedTask

transform_to_image = transforms.ToPILImage()
transform_to_tensor = transforms.ToTensor()


class PatternSynthesizer(Synthesizer):
    pattern_tensor: torch.Tensor = torch.tensor([
        [0., 0., 0.],
        [0., 0., 0.],
        [0., 0., 0.],
    ])
    "Just some random 2D pattern."

    x_top = 3
    "X coordinate to put the backdoor into. 放入后门的x起始坐标"
    y_top = 23
    "Y coordinate to put the backdoor into. 放入后门的y起始坐标"

    x_bot = x_top + pattern_tensor.shape[0]  # 放入后门的x结束坐标
    y_bot = y_top + pattern_tensor.shape[1]  # 放入后门的y结束坐标

    mask_value = -10
    "A tensor coordinate with this value won't be applied to the image. 具有这个值的张量坐标不会被应用到图像上"
    dbas = []

    resize_scale = (5, 10)
    "If the pattern is dynamically placed, resize the pattern. 如果模式是动态放置的，请调整模式的大小"

    mask: torch.Tensor = None
    "A mask used to combine backdoor pattern with the original image. 用于将后门模式与原始图像结合的蒙版"

    pattern: torch.Tensor = None
    "A tensor of the `input.shape` filled with `mask_value` except backdoor. 一个张量，大小为`input.shape`，除了后门部分以外，其余部分填充有 `mask_value`"

    def __init__(self, task: FederatedTask, handcraft, distributed, mal: tuple):  # mal为(0, n_mal)，n_mal为恶意客户端数量
        super().__init__(task)
        self.i_mal = mal[0]
        self.n_mal = mal[1]
        self.make_pattern(task, self.pattern_tensor, self.x_top, self.y_top, handcraft)  # 创建了蒙版和pattern
        if distributed and self.i_mal != 0:
            self.random_break_trigger(task)

    def make_pattern(self, task, pattern_tensor, x_top, y_top, handcraft):
        if isinstance(task, Cifar10FederatedTask):  # 如果是该类型的实例
            trigger_size = (3, 3)
        elif isinstance(task, TinyImagenetFederatedTask):
            trigger_size = (4, 4)
        elif isinstance(task, financeFederatedTask):
            trigger_size = (3, 3)
        if handcraft:  # 如果手动触发器设置为true
            torch.manual_seed(111)  # 设置了随机数种子
            pattern_tensor = torch.rand(trigger_size)  # 创建了一个大小为trigger_size的随机向量
            pattern_tensor = (pattern_tensor * 255).floor() / 255  # 将其中的每个元素限制在 [0, 1] 的范围内
            self.x_bot = x_top + pattern_tensor.shape[0]
            self.y_bot = y_top + pattern_tensor.shape[1]
        else:
            pattern_tensor = torch.zeros(trigger_size)
            self.x_bot = x_top + pattern_tensor.shape[0]
            self.y_bot = y_top + pattern_tensor.shape[1]

        # 创建一个形状为 `input_shape` 的全零张量 `full_image`，并用 `mask_value` 填充整个张量
        full_image = torch.zeros(self.params.input_shape).fill_(self.mask_value)

        x_bot = self.x_bot
        y_bot = self.y_bot

        # mask is 1 when the pattern is presented
        if x_bot >= self.params.input_shape[1] or y_bot >= self.params.input_shape[2]:  # 检查后门的结束位置是否超出了图像范围
            raise ValueError(f'Position of backdoor outside image limits:'
                             f'image: {self.params.input_shape}, but backdoor'
                             f'ends at ({x_bot}, {y_bot})')

        full_image[:, x_top:x_bot, y_top:y_bot] = pattern_tensor  # 将 `pattern_tensor` 放置在 `full_image` 中的指定位置

        # 创建一个蒙版，当full_image的值不等于mask_value时，将蒙版设置为1
        # 张量和标量比较时，会自动将标量转换为与张量相同的数据类型和形状
        self.mask = 1 * (full_image != self.mask_value).to(self.params.device)

        self.pattern = self.task.normalize(full_image).to(self.params.device)  # 归一化处理
        print("---------------", self.pattern.shape)

    def random_break_trigger(self, task):
        x_top, y_top = self.x_top, self.y_top
        i_mal, n_mal = self.i_mal, self.n_mal
        assert (n_mal in [1, 2, 4])  # 断言语句，用于确保变量 `n_mal` 的值在列表 `[1, 2, 4]` 中
        if n_mal == 1:
            if isinstance(task, Cifar10FederatedTask):
                for p in range(3):
                    gx = random.randint(0, 2)
                    gy = random.randint(0, 2)
                    self.mask[:, x_top + gx, y_top + gy] = 0
            elif isinstance(task, TinyImagenetFederatedTask):
                for p in range(9):
                    gx = random.randint(0, 3)
                    gy = random.randint(0, 3)
                    self.mask[:, x_top + gx, y_top + gy] = 0
        elif n_mal == 2:
            if i_mal == 1:
                if isinstance(task, Cifar10FederatedTask):
                    self.mask[:, x_top, y_top] = 0
                    self.mask[:, x_top + 2, y_top] = 0
                    self.mask[:, x_top + 2, y_top] = 0
                    self.mask[:, x_top + 2, y_top + 2] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top, y_top] = 0
                    self.mask[:, x_top + 3, y_top] = 0
                    self.mask[:, x_top, y_top + 3] = 0
                    self.mask[:, x_top + 3, y_top + 3] = 0
            elif i_mal == 2:
                if isinstance(task, Cifar10FederatedTask):
                    self.mask[:, x_top, y_top + 1] = 0
                    self.mask[:, x_top + 2, y_top + 1] = 0
                    self.mask[:, x_top + 1, y_top] = 0
                    self.mask[:, x_top + 1, y_top + 2] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top, y_top + 1:y_top + 3] = 0
                    self.mask[:, x_top + 3, y_top + 1:y_top + 3] = 0
                    self.mask[:, x_top + 1:x_top + 3, y_top] = 0
                    self.mask[:, x_top + 1:x_top + 3, y_top + 3] = 0
            else:
                raise ValueError("out of mal index!")
            print("dba mask:{}:\n".format((i_mal, n_mal)), self.mask[0, 3:7, 23:27])
        elif n_mal == 4:
            if i_mal == 1:
                if isinstance(task, Cifar10FederatedTask):  # 将特定位置的mask设置为零
                    self.mask[:, x_top, y_top] = 0
                    self.mask[:, x_top + 1, y_top] = 0
                    self.mask[:, x_top, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top + 3, y_top:y_top + 4] = 0
                    self.mask[:, x_top:x_top + 4, y_top + 3] = 0
            if i_mal == 2:
                if isinstance(task, Cifar10FederatedTask):
                    self.mask[:, x_top, y_top + 2] = 0
                    self.mask[:, x_top + 1, y_top + 2] = 0
                    self.mask[:, x_top, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top:x_top + 4, y_top] = 0
                    self.mask[:, x_top + 3, y_top:y_top + 4] = 0
            if i_mal == 3:
                if isinstance(task, Cifar10FederatedTask):
                    self.mask[:, x_top + 2, y_top] = 0
                    self.mask[:, x_top + 2, y_top + 1] = 0
                    self.mask[:, x_top + 1, y_top + 0] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top, y_top:y_top + 4] = 0
                    self.mask[:, x_top:x_top + 4, y_top + 3] = 0
            if i_mal == 4:
                if isinstance(task, Cifar10FederatedTask):
                    self.mask[:, x_top + 2, y_top + 2] = 0
                    self.mask[:, x_top + 1, y_top + 2] = 0
                    self.mask[:, x_top + 2, y_top + 1] = 0
                elif isinstance(task, TinyImagenetFederatedTask):
                    self.mask[:, x_top:x_top + 4, y_top] = 0
                    self.mask[:, x_top, y_top:y_top + 4] = 0
            print("dba mask:{}:\n".format((i_mal, n_mal)), self.mask[0, x_top:x_top + 4, y_top:y_top + 4])
        else:
            raise ValueError("Not implement DBA for num of clients out of 1,2,4")

    def synthesize_inputs(self, batch: object, attack_portion: object = None) -> object:
        pattern, mask = self.get_pattern()
        batch.inputs[:attack_portion] = (1 - mask) * batch.inputs[:attack_portion] + mask * pattern

        return

    def synthesize_labels(self, batch, attack_portion=None):
        batch.labels[:attack_portion].fill_(self.params.backdoor_label)

        return

    def get_pattern(self):
        if self.params.backdoor_dynamic_position:  # 如果后门的位置是动态的
            resize = random.randint(self.resize_scale[0], self.resize_scale[1])
            pattern = self.pattern_tensor
            if random.random() > 0.5:
                pattern = functional.hflip(pattern)
            image = transform_to_image(pattern)
            pattern = transform_to_tensor(functional.resize(image, resize, interpolation=0)).squeeze()

            x = random.randint(0, self.params.input_shape[1] - pattern.shape[0] - 1)
            y = random.randint(0, self.params.input_shape[2] - pattern.shape[1] - 1)
            self.make_pattern(pattern, x, y)

        return self.pattern, self.mask


if __name__ == '__main__':
    pass