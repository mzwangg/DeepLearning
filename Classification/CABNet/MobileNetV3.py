import torch
import math
from torch import nn, Tensor
from torch.nn import functional as F

class GlobalAttentionBlock(nn.Sequential):
    def __init__(self, chanel):
        super(GlobalAttentionBlock, self).__init__()
        # 通道注意力模块
        self.chanel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(chanel, chanel, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(chanel, chanel, kernel_size=1, stride=1),
            nn.Sigmoid()
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = x * self.chanel_attention(x)
        # 空间注意力模块
        # 在将原始fature map的chanel数变为1时要注意第一维才是chanel
        x = out * (self.sigmoid(torch.mean(out, dim=1, keepdim=True)))
        return x


class CategoryAttentionBlock(nn.Sequential):
    def __init__(self, input_c, classes, k):
        super(CategoryAttentionBlock, self).__init__()
        self.classes = classes
        self.k = k

        self.layer1 = nn.Sequential(
            nn.Conv2d(input_c, classes * k, kernel_size=1),
            nn.BatchNorm2d(classes * k),
            nn.ReLU()
        )

        self.GMP = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        shape = x.size()
        out1 = self.layer1(x)
        #在reshape是要注意tensor的各维度分别为(batch,chanel,h,w)
        out2 = torch.mean(torch.reshape(self.GMP(out1), [shape[0], self.classes, self.k, 1, 1]), dim=2, keepdim=False)
        out1 = torch.mean(torch.reshape(out1, [shape[0], self.classes, self.k, shape[2], shape[3]]), dim=2,keepdim=False)
        x = x * (torch.mean(out1 * out2, dim=1, keepdim=True))
        return x


class SPPLayer(torch.nn.Module):
    def __init__(self, num_levels):
        super(SPPLayer, self).__init__()
        #num_levels为表示将其按什么尺寸展开的矩阵，第一个数必须为1
        self.num_levels = num_levels

    def forward(self, x):
        num, c, h, w = x.size()  # num:样本数量 c:通道数 h:高 w:宽
        for level in self.num_levels:
            kernel_size = (math.ceil(h / level), math.ceil(w / level))
            stride = (math.ceil(h / level), math.ceil(w / level))
            pooling = (math.floor((kernel_size[0] * level - h + 1) / 2), math.floor((kernel_size[1] * level - w + 1) / 2))
            tensor = F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(num, -1)

            # 展开、拼接
            if (level == self.num_levels[0]):
                x_flatten = tensor.view(num, -1)
            else:
                x_flatten = torch.cat((x_flatten, tensor.view(num, -1)), 1)
        return x_flatten


def _make_divisible(ch, divisor=8, min_ch=None):
    # 确保每个层的通道数都是8的整数倍，会对硬件更友好，提高性能
    if min_ch is None:
        min_ch = divisor
    new_ch = max(min_ch, int(ch + divisor / 2) // divisor * divisor)
    # 确保通道数不会降低超过10%
    if new_ch < 0.9 * ch:
        new_ch += divisor
    return new_ch


# 注意力机制模块
class seBlock(nn.Module):
    def __init__(self, input_c: int, squeeze_factor: int = 4):
        super(seBlock, self).__init__()
        # 确保每个层的通道数都是8的整数倍，会对硬件更友好，提高性能
        squeeze_c = _make_divisible(input_c // squeeze_factor)
        self.fc1 = nn.Conv2d(input_c, squeeze_c, 1)
        self.fc2 = nn.Conv2d(squeeze_c, input_c, 1)

    def forward(self, x: Tensor) -> Tensor:
        scale = F.adaptive_avg_pool2d(x, output_size=(1, 1))
        scale = self.fc1(scale)
        scale = F.relu(scale, inplace=True)
        scale = self.fc2(scale)
        scale = F.hardsigmoid(scale, inplace=True)
        return scale * x


class bneck(nn.Module):
    def __init__(self, input_c, kernel, expanded_c, out_c, use_se, activitor, stride):
        super(bneck, self).__init__()
        self.use_res_connect = (stride == 1 and input_c == out_c)
        layers = []

        # 在第一层bneck层没有这层卷积
        if expanded_c != input_c:
            layers.append(nn.Sequential(
                nn.Conv2d(input_c, expanded_c, kernel_size=1, bias=False),
                nn.BatchNorm2d(expanded_c, eps=0.001, momentum=0.01),
                activitor(inplace=True)
            ))

        # 逐通道卷积
        layers.append(nn.Sequential(
            nn.Conv2d(expanded_c, expanded_c, kernel_size=kernel, stride=stride, padding=(kernel - 1) // 2,
                      groups=expanded_c, bias=False),
            nn.BatchNorm2d(expanded_c, eps=0.001, momentum=0.01),
            activitor(inplace=True)
        ))

        if use_se:
            layers.append(seBlock(expanded_c))

        # 逐点卷积
        layers.append(nn.Sequential(
            nn.Conv2d(expanded_c, out_c, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_c, eps=0.001, momentum=0.01),
            nn.Identity(inplace=True)
        ))

        self.block = nn.Sequential(*layers)

    def forward(self, x: Tensor):
        result = self.block(x)
        if self.use_res_connect:
            result += x

        return result


class mobilenet_v3_large(nn.Module):
    def __init__(self, num_classes=1000):
        super(mobilenet_v3_large, self).__init__()
        layers = []

        # building first layer
        layers.append(nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16, eps=0.001, momentum=0.01),
            nn.Hardswish(inplace=True)
        ))

        layers += [
            bneck(16, 3, 16, 16, False, nn.ReLU, 1),
            bneck(16, 3, 64, 24, False, nn.ReLU, 2),  # C1
            bneck(24, 3, 72, 24, False, nn.ReLU, 1),
            bneck(24, 5, 72, 40, True, nn.ReLU, 2),  # C2
            bneck(40, 5, 120, 40, True, nn.ReLU, 1),
            bneck(40, 5, 120, 40, True, nn.ReLU, 1),
            bneck(40, 3, 240, 80, False, nn.Hardswish, 2),  # C3
            bneck(80, 3, 200, 80, False, nn.Hardswish, 1),
            bneck(80, 3, 184, 80, False, nn.Hardswish, 1),
            bneck(80, 3, 184, 80, False, nn.Hardswish, 1),
            bneck(80, 3, 480, 112, True, nn.Hardswish, 1),
            bneck(112, 3, 672, 112, True, nn.Hardswish, 1),
            bneck(112, 5, 672, 160, True, nn.Hardswish, 2),  # C4
            bneck(160, 5, 960, 160, True, nn.Hardswish, 1),
            bneck(160, 5, 960, 160, True, nn.Hardswish, 1)]

        layers.append(nn.Sequential(
            nn.Conv2d(160, 960, kernel_size=1, bias=False),
            nn.BatchNorm2d(960, eps=0.001, momentum=0.01),
            nn.Hardswish(inplace=True)
        ))

        #将通道减少为原来的一半
        # layers.append(nn.Conv2d(960, 480, kernel_size=1, stride=1))

        #加入CABNet模块
        layers.append(GlobalAttentionBlock(960))
        layers.append(CategoryAttentionBlock(960, num_classes, 5))

        #加入SPP模块
        # layers.append(SPPLayer(num_levels=[1, 4]))

        self.features = nn.Sequential(*layers)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Linear(960, 1280),
                                        nn.Hardswish(inplace=True),
                                        nn.Dropout(p=0.5, inplace=True),
                                        nn.Linear(1280, num_classes))
        self.initial_weights()

    #权重初始化函数
    def initial_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x=self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
