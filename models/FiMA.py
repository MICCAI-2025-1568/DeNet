import math

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


class FineReMambaBlock(nn.Module):

    def __init__(self, n_channels, md_type='e'):
        super().__init__()
        self.mamba = Mamba(d_model=n_channels, d_state=16, d_conv=4, expand=2, md_type=md_type, drop_prob=None)

    def forward(self, x):
        x = self.mamba(x)
        return x


class CoarseReMambaBlock(nn.Module):

    def __init__(self, n_channels, md_type='b'):
        super().__init__()
        self.mamba = Mamba(d_model=n_channels, d_state=16, d_conv=4, expand=2, md_type=md_type, drop_prob=None)

    def forward(self, x):
        x = self.mamba(x)
        return x


class ReMambaBlock(nn.Module):

    def __init__(self, n_channels, fine_md_type='e', coarse_md_type='b'):
        super().__init__()
        self.remamba_f = FineReMambaBlock(n_channels, md_type=fine_md_type)
        self.remamba_c = CoarseReMambaBlock(n_channels, md_type=coarse_md_type)
        self.downsample = nn.Sequential(nn.AdaptiveAvgPool2d(output_size=(1, 1)), nn.Flatten(-3, -1))

    def forward(self, x, prev_x=None):
        B, L, C, H, W = x.shape
        x_f = self.remamba_f(x) + x
        x_c = self.downsample(x).view(B, L, C)
        x_c = self.remamba_c(x_c + prev_x) if prev_x is not None else self.remamba_c(x_c)
        return x_f, x_c


class MultiModalFuserLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()

        self.imu1_acc_k_proj = nn.Linear(3, d_model)
        self.imu1_acc_v_proj = nn.Linear(3, d_model)
        self.imu2_acc_k_proj = nn.Linear(3, d_model)
        self.imu2_acc_v_proj = nn.Linear(3, d_model)
        self.imu3_acc_k_proj = nn.Linear(3, d_model)
        self.imu3_acc_v_proj = nn.Linear(3, d_model)
        self.imu4_acc_k_proj = nn.Linear(3, d_model)
        self.imu4_acc_v_proj = nn.Linear(3, d_model)

        self.imu1_ang_k_proj = nn.Linear(3, d_model)
        self.imu1_ang_v_proj = nn.Linear(3, d_model)
        self.imu2_ang_k_proj = nn.Linear(3, d_model)
        self.imu2_ang_v_proj = nn.Linear(3, d_model)
        self.imu3_ang_k_proj = nn.Linear(3, d_model)
        self.imu3_ang_v_proj = nn.Linear(3, d_model)
        self.imu4_ang_k_proj = nn.Linear(3, d_model)
        self.imu4_ang_v_proj = nn.Linear(3, d_model)

        self.norm_acc = nn.LayerNorm(d_model)
        self.norm_ang = nn.LayerNorm(d_model)

        self.attn_acc = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.attn_ang = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.dropout_acc = nn.Dropout(dropout)
        self.dropout_ang = nn.Dropout(dropout)

        self.linear1 = nn.Linear(d_model * 2, d_model * 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_model * 2, d_model * 2)

        self.norm2 = nn.LayerNorm(d_model * 2)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, accele, angle):
        B, L, C = x.shape
        x = x.permute(1, 0, 2)
        x = x.unsqueeze(-1)
        x = x.view(L * B, C, 1)

        imu1_acc, imu2_acc, imu3_acc, imu4_acc = accele[:, :, 0:3], accele[:, :, 3:6], accele[:, :, 6:9], accele[:, :, 9:12]
        imu1_acc, imu2_acc, imu3_acc, imu4_acc = imu1_acc.permute(1, 0, 2), imu2_acc.permute(1, 0, 2), imu3_acc.permute(1, 0, 2), imu4_acc.permute(1, 0, 2)
        imu1_ang, imu2_ang, imu3_ang, imu4_ang = angle[:, :, 0:3], angle[:, :, 3:6], angle[:, :, 6:9], angle[:, :, 9:12]
        imu1_ang, imu2_ang, imu3_ang, imu4_ang = imu1_ang.permute(1, 0, 2), imu2_ang.permute(1, 0, 2), imu3_ang.permute(1, 0, 2), imu4_ang.permute(1, 0, 2)

        imu1_acc_k, imu2_acc_k, imu3_acc_k, imu4_acc_k = self.imu1_acc_k_proj(imu1_acc), self.imu2_acc_k_proj(imu2_acc), self.imu3_acc_k_proj(imu3_acc), self.imu4_acc_k_proj(imu4_acc)
        imu_acc_k = torch.stack([imu1_acc_k, imu2_acc_k, imu3_acc_k, imu4_acc_k], dim=2)

        imu1_acc_v, imu2_acc_v, imu3_acc_v, imu4_acc_v = self.imu1_acc_v_proj(imu1_acc), self.imu2_acc_v_proj(imu2_acc), self.imu3_acc_v_proj(imu3_acc), self.imu4_acc_v_proj(imu4_acc)
        imu_acc_v = torch.stack([imu1_acc_v, imu2_acc_v, imu3_acc_v, imu4_acc_v], dim=2)

        imu1_ang_k, imu2_ang_k, imu3_ang_k, imu4_ang_k = self.imu1_ang_k_proj(imu1_ang), self.imu2_ang_k_proj(imu2_ang), self.imu3_ang_k_proj(imu3_ang), self.imu4_ang_k_proj(imu4_ang)
        imu_ang_k = torch.stack([imu1_ang_k, imu2_ang_k, imu3_ang_k, imu4_ang_k], dim=2)

        imu1_ang_v, imu2_ang_v, imu3_ang_v, imu4_ang_v = self.imu1_ang_v_proj(imu1_ang), self.imu2_ang_v_proj(imu2_ang), self.imu3_ang_v_proj(imu3_ang), self.imu4_ang_v_proj(imu4_ang)
        imu_ang_v = torch.stack([imu1_ang_v, imu2_ang_v, imu3_ang_v, imu4_ang_v], dim=2)

        imu_acc_k = imu_acc_k.view(L * B, imu_acc_k.shape[-2], C)
        weight_acc = torch.bmm(imu_acc_k, x) / math.sqrt(C)
        weight_acc = F.softmax(weight_acc, dim=1)
        weight_acc = weight_acc.permute(0, 2, 1)

        imu_acc_v = imu_acc_v.view(L * B, imu_acc_v.shape[-2], C)
        imu_acc_emb = torch.bmm(weight_acc, imu_acc_v)
        imu_acc_emb = imu_acc_emb.view(L, B, C)
        imu_acc_emb = self.norm_acc(imu_acc_emb)

        imu_ang_k = imu_ang_k.view(L * B, imu_ang_k.shape[-2], C)
        weight_ang = torch.bmm(imu_ang_k, x) / math.sqrt(C)
        weight_ang = F.softmax(weight_ang, dim=1)
        weight_ang = weight_ang.permute(0, 2, 1)

        imu_ang_v = imu_ang_v.view(L * B, imu_ang_v.shape[-2], C)
        imu_ang_emb = torch.bmm(weight_ang, imu_ang_v)
        imu_ang_emb = imu_ang_emb.view(L, B, C)
        imu_ang_emb = self.norm_ang(imu_ang_emb)

        x = x.view(L, B, C)
        imu_acc_emb = self.attn_acc(x, imu_acc_emb, imu_acc_emb)[0]
        x = x + self.dropout_acc(imu_acc_emb)
        imu_ang_emb = self.attn_ang(x, imu_ang_emb, imu_ang_emb)[0]
        x = torch.cat([x, self.dropout_ang(imu_ang_emb)], dim=2)

        x = self.norm2(x)
        _x = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(_x)

        return x.permute(1, 0, 2)


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

        self.mamba1 = ReMambaBlock(self.channels[0])
        self.mamba2 = ReMambaBlock(self.channels[1])
        self.mamba3 = ReMambaBlock(self.channels[2])
        self.mamba4 = ReMambaBlock(self.channels[3])

        self.downsample = nn.Sequential(
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(-3, -1)
        )

        self.fuse = MultiModalFuserLayer(d_model=self.channels[3], nhead=4)

        self.mamba5 = CoarseReMambaBlock(self.channels[3] * 2)
        self.out = nn.Linear(self.channels[3] * 2, 6)

    def forward(self, x, accele, angle):
        x = 2 * (x / 255.0) - 1.0
        B, L, C, H, W = x.shape
        x = x.view(B * L, C, H, W)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x).view(B, L, *x.shape[-3:])
        x, xc = self.mamba1(x)
        xc1 = F.interpolate(xc, self.channels[-1], mode='linear', align_corners=True)

        x = self.layer2(x.view(B * L, *x.shape[-3:]))
        x = x.view(B, L, *x.shape[-3:]);
        xc = F.interpolate(xc, self.channels[1], mode='linear', align_corners=True)
        x, xc = self.mamba2(x, xc)
        xc2 = F.interpolate(xc, self.channels[-1], mode='linear', align_corners=True)

        x = self.layer3(x.view(B * L, *x.shape[-3:]))
        x = x.view(B, L, *x.shape[-3:]);
        xc = F.interpolate(xc, self.channels[2], mode='linear', align_corners=True)
        x, xc = self.mamba3(x, xc)
        xc3 = F.interpolate(xc, self.channels[-1], mode='linear', align_corners=True)

        x = self.layer4(x.view(B * L, *x.shape[-3:]))
        x = x.view(B, L, *x.shape[-3:]);
        xc = F.interpolate(xc, self.channels[3], mode='linear', align_corners=True)
        x, xc4 = self.mamba4(x, xc)

        x = self.downsample(x.view(B * L, *x.shape[-3:])) + xc1 + xc2 + xc3 + xc4

        x = self.fuse(x.view(B, L, x.shape[-1]), accele, angle)
        x = self.mamba5(x) + x
        x = self.out(x)

        return x


class FiMA(models.BaseModel):

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
