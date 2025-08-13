import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Iterable, Tuple

def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.InstanceNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.InstanceNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class CombineMRIWithTemperature(nn.Module):
    def __init__(self, scale_factor, channel_mri, channel_temp):
        super(CombineMRIWithTemperature, self).__init__()
        self.scale_factor = scale_factor
        self.conv = nn.Conv3d(channel_mri, channel_temp, 1, 1)
        self.bn_feat = nn.InstanceNorm3d(channel_temp)
        self.relu_feat = nn.ReLU(inplace=True)
        # 可学习参数（自动梯度计算）
        self.w_high = nn.Parameter(torch.tensor(0.1))  # 高温权重
        self.w_mid = nn.Parameter(torch.tensor(0.1))  # 中温权重

    @property
    def weights(self):
        """动态计算约束后的权重"""
        # 使用Sigmoid约束参数范围
        w_high = torch.tanh(self.w_high)  # 确保≤0.3
        w_mid = torch.tanh(self.w_mid)
        return w_high, w_mid

    def forward(self, mri_feat, temp, temp_feat):
        # 温度特征插值
        temp_attention = F.interpolate(temp, scale_factor=(self.scale_factor, self.scale_factor, self.scale_factor), mode='trilinear', align_corners=False)

        # # 使用掩膜引导特征学习
        mri_feat_high = mri_feat * temp_attention
        mri_feat_high = self.relu_feat(self.bn_feat(self.conv(mri_feat_high)))

        middle_temp_mask = torch.exp(-((temp_attention - 0.27) ** 2) / 0.02)
        mri_feat_middle = mri_feat * middle_temp_mask
        mri_feat_middle = self.relu_feat(self.bn_feat(self.conv(mri_feat_middle)))

        w_high, w_mid = self.weights

        # 加权融合
        fused = temp_feat + (w_high * mri_feat_high) + (w_mid * mri_feat_middle)

        return fused

    def get_current_weights(self):
        """获取当前有效权重"""
        with torch.no_grad():
            w_high, w_mid = self.weights
            return {
                "w_high": w_high.item(),
                "w_mid": w_mid.item(),
            }
class DecoderSeparable(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2, bias=False)
        # depthwise 3x3x3
        self.depthwise = nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1,
                                   groups=out_channels, bias=False)
        # pointwise 1x1x1
        self.pointwise = nn.Conv3d(out_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.InstanceNorm3d(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.up(x)
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.norm(x)
        return self.act(x)

class PAZNet(nn.Module):
    def __init__(self, training=True, no_cuda=False):
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.training = training
        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False)
        self.bn1 = nn.InstanceNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(BasicBlock, 64, 1)
        self.layer2 = self._make_layer(BasicBlock, 128, 1, stride=2)

        # 温度先验的编码过程
        self.down_conv1 = nn.Sequential(nn.Conv3d(1, 16, 2, 2), nn.Conv3d(16, 64, 3, 1, 1), nn.InstanceNorm3d(64), nn.ReLU(inplace=True))
        self.down_conv2 = nn.Sequential(nn.Conv3d(64, 64, 2, 2), nn.InstanceNorm3d(64), nn.ReLU(inplace=True))
        self.down_conv3 = nn.Sequential(nn.Conv3d(64, 128, 2, 2), nn.InstanceNorm3d(128), nn.ReLU(inplace=True))

        self.combine_layer1 = CombineMRIWithTemperature(0.125, 128, 128)
        self.combine_layer2 = CombineMRIWithTemperature(0.25, 64, 64)
        self.combine_layer3 = CombineMRIWithTemperature(0.5, 64, 64)

        # 解码器
        self.decoder1 = DecoderSeparable(128, 64)
        self.decoder2 = DecoderSeparable(128, 64)
        self.decoder3 = DecoderSeparable(128, 16)

        # 最终输出卷积层
        self.map3 = nn.Sequential(
            nn.Conv3d(16, 1, 1, 1),
            nn.Sigmoid()
        )

        # 64*64 尺度下的映射
        self.map2 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(2, 2, 2), mode='trilinear'),
            nn.Sigmoid()
        )

        # 32*32 尺度下的映射
        self.map1 = nn.Sequential(
            nn.Conv3d(64, 1, 1, 1),
            nn.Upsample(scale_factor=(4, 4, 4), mode='trilinear'),
            nn.Sigmoid()
        )
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.inplanes,
                    planes * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False), nn.InstanceNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        # 在每个块的输出后添加 Dropout3d 层 CRD
        layers.append(nn.Dropout3d(p=0.2))  # 20% 的通道丢弃

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv3d, nn.ConvTranspose3d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if getattr(m, "bias", None) is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, mri, temp):
        x = self.conv1(mri)  # [6, 64, 40, 40, 40]
        x = self.bn1(x)  # [6, 64, 40, 40, 40]
        x0 = self.relu(x)  # [6, 64, 40, 40, 40]
        x = self.maxpool(x0)  # [6, 64, 20, 20, 20]
        x1 = self.layer1(x)  # [6, 64, 20, 20, 20]
        x2 = self.layer2(x1)  # [6, 128, 10, 10, 10]

        y1 = self.down_conv1(temp)  # [6, 32, 40, 40, 40]
        y2 = self.down_conv2(y1)  # [6, 64, 20, 20, 20]
        y3 = self.down_conv3(y2)  # [6, 256, 10, 10, 10]
        # Step 4: 解码
        xy1 = self.combine_layer1(x2, temp, y3)  # [6, 128, 10, 10, 10]
        z = self.decoder1(xy1)  # [6, 64, 20, 20, 20]
        z = F.dropout(z, 0.3, self.training)
        output1 = self.map1(z)  # [6, 1, 80, 80, 80]

        xy2 = self.combine_layer2(x1, temp, y2)  # [6, 64, 20, 20, 20]
        z = self.decoder2(torch.cat([z, xy2], dim=1))  # [6, 64, 40, 40, 40]
        z = F.dropout(z, 0.3, self.training)
        output2 = self.map2(z)  # [6, 1, 80, 80, 80]

        xy3 = self.combine_layer3(x0, temp, y1)  # [6, 64, 40, 40, 40]
        z = self.decoder3(torch.cat([z, xy3], dim=1))  # [6, 16, 80, 80, 80]
        z = F.dropout(z, 0.3, self.training)
        output3 = self.map3(z)  # [6, 1, 80, 80, 80]
        if self.training is True:
            return output3, output2, output1
        else:
            return output3

def generate_model(training: bool = True,
                   no_cuda: bool = False,
                   gpu_id = [0],
                   phase: str = 'train',
                   pretrain_path: str = None
                   ) -> Tuple[nn.Module, Dict[str, Iterable[nn.Parameter]]]:

    model = PAZNet(training=training)

    if not no_cuda:
        if len(gpu_id) > 1:
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=gpu_id)
            net_dict = model.state_dict()
        else:
            import os
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            model = model.cuda()
            model = nn.DataParallel(model, device_ids=None)
            net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if phase != 'test' and pretrain_path:
        print('loading pretrained model {}'.format(pretrain_path))
        pretrain = torch.load(pretrain_path, weights_only=True)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}
        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)
        print("-------- pre-train model load successfully --------")

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in ['attention_layer', 'decoder', 'map', 'up_conv', 'down_conv', 'combine_layer',
                               'initial_conv', 'initial_ln']:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {'base_parameters': base_parameters,
                      'new_parameters': new_parameters}
        new_params_count = sum(p.numel() for p in new_parameters)
        print(f"New parameters: {new_params_count}")
        base_params_count = sum(p.numel() for p in base_parameters)
        print(f"Base parameters: {base_params_count}")
        return model, parameters

    return model, model.parameters()