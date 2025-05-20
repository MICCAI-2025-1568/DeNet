import os

import numpy as np
import torch
import torch.nn.functional as F

import configs
import datasets
import utils


class IMU4(datasets.BaseDataset):

    @staticmethod
    def more(cfg):
        cfg.source.elements = cfg.source.width * cfg.source.height * cfg.source.channel
        cfg.paths.source = configs.env.getdir(cfg.paths.source)
        cfg.device = torch.device('cpu')
        return cfg

    def load(self):
        source_data_path = os.path.join(self.cfg.paths.source, self.__class__.__name__ + '_' + self.cfg.data_mode + '.npy')
        npy = np.load(source_data_path, allow_pickle=True)[()]
        source_data, self.source_idx = npy['data'], npy['idx']
        return {'source': source_data}, len(self.source_idx)

    def preprocessing(self, tp):
        tp = tp.view(-1, 3, 3)
        pall = torch.cat([tp, 2 * tp[:, 0:1, :] - tp[:, 1:2, :], 2 * tp[:, 0:1, :] - tp[:, 2:3, :]], dim=1)
        min_loca = torch.min(pall.reshape(-1, 3), dim=0)[0]
        tp = tp - min_loca.unsqueeze(0).unsqueeze(0)
        td = utils.functional.series_to_dof(tp)
        tp = tp.view(-1, 9)
        return tp, td

    def __getitem__(self, index):
        idx_obj, idx_data = self.source_idx[self.get_idx(index)]
        data = self.data['source'][idx_obj][idx_data - 1]

        source, series, accele, imu_series, imu_accele = data['frame'], data['series'], data['acc'], data['imu_series'], data['imu_acc']
        info = torch.tensor([len(source)])

        flip = torch.rand(1)[0] > 0.5
        frame_rate = torch.randint(self.cfg.frame_rate[0], self.cfg.frame_rate[1] + 1, (1,))[0]
        if not flip:
            index_select = torch.arange(0, len(source), frame_rate, dtype=torch.long, device=source.device)
        else:
            index_select = torch.arange(len(source) - 1, -1, -frame_rate, dtype=torch.long, device=source.device)

        if len(index_select) >= self.cfg.series_length[0]:
            series_length = torch.randint(self.cfg.series_length[0], min(len(index_select), self.cfg.series_length[1]) + 1, (1,))[0]
            series_start = torch.randint(0, len(index_select) - series_length + 1, (1,))[0]
        else:
            series_length, series_start = len(index_select), 0
        index_select = index_select[series_start:series_start + series_length]

        source = torch.index_select(source, 0, index_select)
        series = torch.index_select(series, 0, index_select)
        accele = torch.index_select(accele, 0, index_select)
        imu_series = torch.index_select(imu_series, 1, index_select)
        imu_accele = torch.index_select(imu_accele, 1, index_select)
        if flip:
            accele = -accele
            imu_accele = -imu_accele

        series, dof = self.preprocessing(series)
        imu_angle = []
        for i in range(len(imu_series)):
            imu_angle.append(self.preprocessing(imu_series[i])[1][:, 3:])
        imu_angle = torch.stack(imu_angle, dim=0)

        source = source.unsqueeze(1)
        target = torch.cat([F.pad(dof, (0, 0, 0, 1)), series], dim=-1)

        source = source.type(torch.float32).to(self.cfg.device)
        target = target.type(torch.float32).to(self.cfg.device)
        accele = accele.type(torch.float32).to(self.cfg.device)
        imu_angle = imu_angle.type(torch.float32).to(self.cfg.device)
        imu_accele = imu_accele.type(torch.float32).to(self.cfg.device)
        imu_series = imu_series.type(torch.float32).to(self.cfg.device)

        sample_dict = {
            'source': source, 'target': target,
            'accele': accele,
            'imu_angle': imu_angle,
            'imu_accele': imu_accele,
            'imu_series': imu_series,
            'flip': flip,
            'frame_rate': frame_rate,
            'series_start': series_start,
            'series_length': series_length,
            'info': info
        }

        return sample_dict, index
