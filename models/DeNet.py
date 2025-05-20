import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

import models
import utils
from models.mamba.mamba_ssm.modules.mamba_simple import Mamba


class FeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation=nn.ReLU(), is_gated: bool = False,
                 bias1: bool = True, bias2: bool = True, bias_gate: bool = True):
        super().__init__()
        self.layer1 = nn.Linear(d_model, d_ff, bias=bias1)
        self.layer2 = nn.Linear(d_ff, d_model, bias=bias2)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation
        self.is_gated = is_gated
        if is_gated:
            self.linear_v = nn.Linear(d_model, d_ff, bias=bias_gate)

    def forward(self, x: torch.Tensor):
        g = self.activation(self.layer1(x))
        if self.is_gated:
            x = g * self.linear_v(x)
        else:
            x = g
        x = self.dropout(x)
        return self.layer2(x)


class LocalMambaBlock(nn.Module):

    def __init__(self, n_channels, md_type='e'):
        super().__init__()
        self.mamba = Mamba(d_model=n_channels, d_state=16, d_conv=4, expand=2, md_type=md_type, drop_prob=None)

    def forward(self, x):
        x = self.mamba(x)
        return x


class GlobalMamba(nn.Module):

    def __init__(self, n_channels, md_type='b'):
        super().__init__()
        self.mamba = Mamba(d_model=n_channels, d_state=16, d_conv=4, expand=2, md_type=md_type, drop_prob=None)

    def forward(self, x):
        x = self.mamba(x)
        return x


class GlobalMambaBlock(nn.Module):

    def __init__(self, n_channels, coarse_md_type='b'):
        super().__init__()
        self.mamba_c = GlobalMamba(n_channels, md_type=coarse_md_type)
        self.downsample = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(-3, -1))

    def forward(self, x):
        B, L, C, H, W = x.shape
        x_c = self.downsample(x).view(B, L, C)
        x_c = self.mamba_c(x_c)
        return x_c


class AxisFuserLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        self.acc_proj = nn.Linear(12, d_model)
        self.ang_proj = nn.Linear(12, d_model)
        self.acc_temporal = GlobalMamba(d_model)

        self.norm = nn.LayerNorm(d_model)
        self.norm_acc = nn.LayerNorm(d_model)
        self.norm_ang = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

    def forward(self, x, accele, angle):
        B, L, C = x.shape
        acc_emb = self.acc_proj(accele)
        ang_emb = self.ang_proj(angle)
        acc_emb = self.acc_temporal(acc_emb)
        x = self.norm(x)
        acc_emb = self.norm_acc(acc_emb)
        ang_emb = self.norm_ang(ang_emb)
        x = torch.cat((x, acc_emb, ang_emb), dim=1)
        x = self.attn(x, x, x)[0]
        x = torch.cat((x[:, :L], x[:, L:2 * L], x[:, 2 * L:]), dim=2)
        return x


class MoE(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.experts = nn.ModuleList([
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels),
            nn.Linear(in_channels, out_channels)
        ])
        self.gate = nn.Sequential(
            nn.Linear(in_channels, 128),
            nn.Dropout(0.1),
            nn.Linear(128, 256),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 8)
        )

    def forward(self, x):
        gating_score = F.softmax(self.gate(x), dim=-1)
        for i, expert in enumerate(self.experts):
            expertout = expert(x)
            if i == 0:
                output = expertout * gating_score[:, :, i:i + 1]
            else:
                output += expertout * gating_score[:, :, i:i + 1]
        return output


class Structure(nn.Module):

    def __init__(self):
        super().__init__()
        self.channels = [64, 128, 256, 512]
        resnet = torchvision.models.resnet18(weights=torchvision.models.ResNet18_Weights.IMAGENET1K_V1)
        conv1_weight = resnet.conv1.weight.clone()
        resnet.conv1 = nn.Conv2d(2, 64, kernel_size=7, stride=2, padding=3, bias=False)
        resnet.conv1.weight = torch.nn.Parameter(conv1_weight[:, :2, :, :])

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        self.g_temporal1 = GlobalMambaBlock(self.channels[0])
        self.g_temporal2 = GlobalMambaBlock(self.channels[1])
        self.g_temporal3 = GlobalMambaBlock(self.channels[2])
        self.g_temporal4 = GlobalMambaBlock(self.channels[3])

        self.l_temporal1 = LocalMambaBlock(self.channels[0])
        self.l_temporal2 = LocalMambaBlock(self.channels[1])
        self.l_temporal3 = LocalMambaBlock(self.channels[2])
        self.l_temporal4 = LocalMambaBlock(self.channels[3])

        self.con2_1 = nn.Conv2d(192, 128, kernel_size=1, stride=1, padding=0, bias=False)
        self.con2_2 = nn.Conv2d(384, 256, kernel_size=1, stride=1, padding=0, bias=False)
        self.con2_3 = nn.Conv2d(768, 512, kernel_size=1, stride=1, padding=0, bias=False)
        self.downsample = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(-3, -1), )

        self.down_mam = nn.MaxPool2d(kernel_size=7, stride=2, padding=3)

        self.fuse_dis = AxisFuserLayer(d_model=self.channels[3], nhead=16)
        self.fuse_ang = nn.Linear(12, 512)

        self.con11_down = nn.Conv1d(1536, 512, kernel_size=1)

        self.mamba5_dis = GlobalMamba(self.channels[3] * 3)
        self.mamba5_ang = GlobalMamba(self.channels[3] * 2)

        self.disMoE = MoE(self.channels[3] * 3, self.channels[3] * 3)
        self.angMoE = MoE(self.channels[3] * 2, self.channels[3] * 2)

        self.out_dis = nn.Linear(self.channels[3] * 3, 3)
        self.out_ang = nn.Linear(self.channels[3] * 2, 3)

    def forward(self, x, accele, angle):
        x = 2 * (x / 255.0) - 1.0
        B, L, C, H, W = x.shape
        x = x.view(B * L, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x_final = self.layer4(x3)

        x1 = x1.view(B, L, *x1.shape[-3:])
        x_c = self.g_temporal1(x1)
        xc1 = F.interpolate(x_c, self.channels[-1], mode='linear', align_corners=True)

        x2 = x2.view(B, L, *x2.shape[-3:])
        x_c = self.g_temporal2(x2)
        xc2 = F.interpolate(x_c, self.channels[-1], mode='linear', align_corners=True)

        x3 = x3.view(B, L, *x3.shape[-3:])
        x_c = self.g_temporal3(x3)
        xc3 = F.interpolate(x_c, self.channels[-1], mode='linear', align_corners=True)

        x4 = x_final.view(B, L, *x_final.shape[-3:])
        x_c = self.g_temporal4(x4)
        xc4 = F.interpolate(x_c, self.channels[-1], mode='linear', align_corners=True)

        x2 = x2.view(B * L, *x2.shape[-3:])
        x3 = x3.view(B * L, *x3.shape[-3:])
        x4 = x4.view(B * L, *x4.shape[-3:])
        x2_1 = self.l_temporal1(x1)

        x2_1 = x2_1.view(B * L, *x2_1.shape[-3:])
        x = torch.cat((self.down_mam(x2_1), x2), dim=-3)
        x = self.con2_1(x)
        x = x.view(B, L, *x.shape[-3:])
        x2_2 = self.l_temporal2(x)

        x2_2 = x2_2.view(B * L, *x2_2.shape[-3:])
        x = torch.cat((self.down_mam(x2_2), x3), dim=-3)
        x = self.con2_2(x)
        x = x.view(B, L, *x.shape[-3:])
        x2_3 = self.l_temporal3(x)

        x2_3 = x2_3.view(B * L, *x2_3.shape[-3:])
        x = torch.cat((self.down_mam(x2_3), x4), dim=-3)
        x = self.con2_3(x)
        x = x.view(B, L, *x.shape[-3:])
        x2_4 = self.l_temporal4(x)

        x_f = self.downsample(x2_4.view(B * L, *x2_4.shape[-3:])).view(B, L, -1)
        x_c = xc1 + xc2 + xc3 + xc4
        x_S = self.downsample(x_final).view(B, L, -1)

        x = torch.cat((x_f, x_c, x_S), dim=2).permute(0, 2, 1)
        x = self.con11_down(x).permute(0, 2, 1)

        x_dis = self.fuse_dis(x.view(B, L, x.shape[-1]), accele, angle)
        x_ang = torch.cat((x, self.fuse_ang(angle)), dim=2)

        x_dis = self.mamba5_dis(x_dis)
        x_ang = self.mamba5_ang(x_ang)

        x_dis = self.disMoE(x_dis) + x_dis
        x_ang = self.angMoE(x_ang) + x_ang

        output_dis = self.out_dis(x_dis)
        output_ang = self.out_ang(x_ang)

        return torch.cat((output_dis, output_ang), dim=-1)


class DeNet(models.BaseModel):

    def __init__(self, cfg, data_cfg, run, **kwargs):
        super().__init__(cfg, data_cfg, run, **kwargs)
        self.structure = Structure().to(self.device)
        self.optimizer = torch.optim.Adam(self.structure.parameters(), lr=self.run.lr, betas=self.run.betas, weight_decay=self.run.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.run.step_size, gamma=self.run.gamma)

    def correlation_loss(self, labels, outputs):
        x = outputs.flatten()
        y = labels.flatten()
        xy = x * y
        mean_xy = torch.mean(xy)
        mean_x = torch.mean(x)
        mean_y = torch.mean(y)
        cov_xy = mean_xy - mean_x * mean_y

        var_x = torch.sum((x - mean_x) ** 2 / x.shape[0]) + 1e-6
        var_y = torch.sum((y - mean_y) ** 2 / y.shape[0]) + 1e-6

        corr_xy = cov_xy / torch.sqrt(var_x * var_y)

        loss = 1 - corr_xy
        return loss

    def criterion(self, real_target, fake_target):
        real_dist, real_angle = real_target.split([3, self.data_cfg.target.elements - 12], dim=-1)
        fake_dist, fake_angle = fake_target.split([3, self.data_cfg.target.elements - 12], dim=-1)

        loss_dist = F.l1_loss(real_dist, fake_dist)
        loss_angle = F.l1_loss(real_angle, fake_angle)
        loss_corr = self.correlation_loss(real_target, fake_target)

        loss_dict = {'loss_dist': loss_dist, 'loss_angle': loss_angle, 'loss_corr': loss_corr}

        return loss_dict

    def train(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device)
        angle = sample_dict['imu_angle'].to(self.device) if self.cfg.angle else None
        accele = sample_dict['imu_accele'][:, :, 1:].to(self.device) if self.cfg.accele else None

        real_target = real_target[:, :-1, :-9]
        real_target[:, :, 3:] = real_target[:, :, 3:] * 100
        if self.cfg.angle:
            angle = angle * 100

        self.structure.train()
        self.optimizer.zero_grad()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)
        accele = torch.cat([accele[:, i] for i in self.cfg.imu_ids], dim=-1) if self.cfg.accele else None
        angle = torch.cat([angle[:, i] for i in self.cfg.imu_ids], dim=-1) if self.cfg.angle else None

        fake_target = self.structure(input, accele=accele, angle=angle)

        losses = self.criterion(real_target, fake_target)
        loss = sum(losses.values())
        loss.backward()
        self.optimizer.step()
        self.scheduler.step(epoch_info['epoch'])

        return {'loss': loss, **losses}

    def test(self, epoch_info, sample_dict):
        real_source = sample_dict['source'].to(self.device)
        real_target = sample_dict['target'].to(self.device).squeeze(0)
        angle = sample_dict['imu_angle'].to(self.device) if self.cfg.angle else None
        accele = sample_dict['imu_accele'][:, :, 1:].to(self.device) if self.cfg.accele else None

        real_series = real_target[:, -9:].view(-1, 3, 3)
        if self.cfg.angle:
            angle = angle * 100

        self.structure.eval()
        input = torch.cat([real_source[:, :-1, ...], real_source[:, 1:, ...]], dim=2)
        accele = torch.cat([accele[:, i] for i in self.cfg.imu_ids], dim=-1) if self.cfg.accele else None
        angle = torch.cat([angle[:, i] for i in self.cfg.imu_ids], dim=-1) if self.cfg.angle else None

        fake_gaps = self.structure(input, accele=accele, angle=angle)

        fake_gaps = fake_gaps[0, :, :]
        fake_gaps[:, 3:] /= 100

        fake_series = utils.functional.dof_to_series(real_series[0:1, :, :], fake_gaps.unsqueeze(0)).squeeze(0)
        losses = utils.metric.get_metric(real_series, fake_series)

        return losses

    def test_return_hook(self, epoch_info, return_all):
        return_info = {}
        for key, value in return_all.items():
            return_info[key] = np.sum(value) / epoch_info['batch_per_epoch']
        if return_info:
            self.logger.info_scalars('Test Epoch: {}\t', (epoch_info['epoch'],), return_info)
        return return_all
