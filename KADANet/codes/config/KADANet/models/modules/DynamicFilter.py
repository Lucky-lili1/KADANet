import torch
import torch.nn as nn
from timm.layers.helpers import to_2tuple

class StarReLU(nn.Module):
    def __init__(self, scale_value=1.0, bias_value=0.0,
                 scale_learnable=True, bias_learnable=True,
                 mode=None, inplace=False):
        super().__init__()  # 调用父类nn.Module的构造函数
        self.inplace = inplace  # 是否进行原地操作
        self.relu = nn.ReLU(inplace=inplace)  # 定义ReLU激活层
        # 定义可学习的缩放参数s，默认值为scale_value，是否需要梯度更新由scale_learnable决定
        self.scale = nn.Parameter(scale_value * torch.ones(1),
                              requires_grad=scale_learnable)
        # 定义可学习的偏置参数b，默认值为bias_value，是否需要梯度更新由bias_learnable决定
        self.bias = nn.Parameter(bias_value * torch.ones(1),
                             requires_grad=bias_learnable)

    def forward(self, x):
        """
        前向传播函数，计算StarReLU激活后的输出。
        """
        return self.scale * self.relu(x) ** 2 + self.bias  # 应用StarReLU公式

# 定义多层感知机（MLP）类，常用于MetaFormer系列模型
class Mlp(nn.Module):
    """
    MLP模块，类似于Transformer、MLP-Mixer等模型中使用的多层感知机。
    大部分代码来源于timm库。
    """
    def __init__(self, dim, mlp_ratio=4, out_features=None, act_layer=StarReLU, drop=0.,
                 bias=False, **kwargs):
        super().__init__()  # 调用父类nn.Module的构造函数
        in_features = dim  # 输入特征维度
        out_features = out_features or in_features  # 输出特征维度，默认与输入相同
        hidden_features = int(mlp_ratio * in_features)  # 隐藏层特征维度
        drop_probs = to_2tuple(drop)  # 将dropout概率转换为二元组

        # 第一个全连接层，将输入维度映射到隐藏层维度
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.act = act_layer()  # 激活函数层，使用StarReLU
        self.drop1 = nn.Dropout(drop_probs[0])  # 第一个dropout层
        # 第二个全连接层，将隐藏层维度映射回输出维度
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)
        self.drop2 = nn.Dropout(drop_probs[1])  # 第二个dropout层

    def forward(self, x):
        """
        前向传播函数，依次通过全连接层、激活函数和dropout层。
        """
        x = self.fc1(x)  # 第一个全连接层
        x = self.act(x)  # 激活函数
        x = self.drop1(x)  # 第一个dropout层
        x = self.fc2(x)  # 第二个全连接层
        x = self.drop2(x)  # 第二个dropout层
        return x  # 返回输出
def resize_complex_weight(origin_weight, new_h, new_w):
        h, w, num_heads = origin_weight.shape[0:3]  # size, w, c, 2
        origin_weight = origin_weight.reshape(1, h, w, num_heads * 2).permute(0, 3, 1, 2)
        new_weight = torch.nn.functional.interpolate(
            origin_weight,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=True
        ).permute(0, 2, 3, 1).reshape(new_h, new_w, num_heads, 2)
        return new_weight
class DynamicFilter1(nn.Module):
    def __init__(self, dim, expansion_ratio=2, reweight_expansion_ratio=.25,
                 act1_layer=StarReLU, act2_layer=nn.Identity,
                 bias=False, num_filters=4, size=64, weight_resize=False,
                 **kwargs):
        super().__init__()
        size = to_2tuple(size)
        self.size = size[0]
        self.filter_size = size[1] // 2 + 1
        self.num_filters = num_filters
        self.dim = dim
        self.med_channels = int(expansion_ratio * dim)
        self.weight_resize = weight_resize
        self.pwconv1 = nn.Linear(dim, self.med_channels, bias=bias)
        self.act1 = act1_layer()
        self.reweight = Mlp(dim, reweight_expansion_ratio, num_filters * self.med_channels)
        self.complex_weights = nn.Parameter(
            torch.randn(self.size, self.filter_size, num_filters, 2,
                        dtype=torch.float32) * 0.02)
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(self.med_channels, dim, bias=bias)

    def forward(self, x):
        # print('x调整之前的',x.shape)
        # (B, C, H, W) → (B, H, W, C)

        B, H, W, _ = x.shape
        #print('DynamicFilter的x输入.shape', x.shape)

        routeing = self.reweight(x.mean(dim=(1, 2))).view(B, self.num_filters,
                                                          -1).softmax(dim=1)
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.to(torch.float32)
        x = torch.fft.rfft2(x, dim=(1, 2), norm='ortho')
        #print('傅里叶之后的x输入.shape', x.shape)

        if self.weight_resize:
            complex_weights = resize_complex_weight(self.complex_weights, x.shape[1],x.shape[2])
            complex_weights = torch.view_as_complex(complex_weights.contiguous())
        else:
            complex_weights = torch.view_as_complex(self.complex_weights)
        routeing = routeing.to(torch.complex64)
        weight = torch.einsum('bfc,hwf->bhwc', routeing, complex_weights)
        if self.weight_resize:
            weight = weight.view(-1, x.shape[1], x.shape[2], self.med_channels)
        else:
            weight = weight.view(-1, self.size, self.filter_size, self.med_channels)
        #print('最终的weight输入.shape', weight.shape)
        x = x * weight
        x = torch.fft.irfft2(x, s=(H, W), dim=(1, 2), norm='ortho')

        x = self.act2(x)
        x = self.pwconv2(x)
        return x
        #return x.permute(0, 3, 1, 2)  # (B, H, W, C) → (B, C, H, W)
