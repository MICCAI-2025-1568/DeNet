import numpy as np
import pyvista as pv
import torch

import utils
from models.DeNet import Structure


def read_data(id_case, path_base='cases', data_mode='Arm'):
    path_case = f'{path_base}/{data_mode}_{id_case}.npy'
    data_case = np.load(path_case, allow_pickle=True)[()]
    return data_case


def infer(id_data=1, path_weight='weights/denet_Arm.pth', dtype=torch.float32, device='cuda:0'):
    structure = Structure().to(device)
    structure.load_state_dict(torch.load(path_weight, map_location=device))
    structure.eval()

    data = read_data(id_data)
    with torch.no_grad():
        source = torch.from_numpy(data['source']).type(dtype).to(device).unsqueeze(0).unsqueeze(2)
        accele = torch.from_numpy(data['accele']).type(dtype).to(device).unsqueeze(0)
        angle = torch.from_numpy(data['angle']).type(dtype).to(device).unsqueeze(0)
        dof = torch.from_numpy(data['dof']).type(dtype).to(device)

        input = torch.cat([source[:, :-1, ...], source[:, 1:, ...]], dim=2)
        accele = torch.cat([accele[:, i] for i in range(4)], dim=-1)
        angle = torch.cat([angle[:, i] for i in range(4)], dim=-1)
        angle = angle * 100

        fake_dof = structure(input, accele=accele, angle=angle)
        fake_dof = fake_dof[0, :, :]
        fake_dof[:, 3:] /= 100

    zero_plane = utils.functional.get_pixelated_zero_plane(248, 260, dtype=torch.float32, device=device)
    real = utils.functional.dof_to_series(zero_plane, dof.unsqueeze(0)).squeeze(0)
    pred = utils.functional.dof_to_series(zero_plane, fake_dof.unsqueeze(0)).squeeze(0)

    volume, bias = utils.reconstruction.reco(source.squeeze(), pred)
    real, pred = real - bias, pred - bias

    pv.set_plot_theme('document')
    p = pv.Plotter()

    labels = [('Ground Truth', 'red'), ('DeNet', 'blue')]
    utils.render.pv_series(p, real.cpu().numpy(), color=labels[0][1])
    utils.render.pv_series(p, pred.cpu().numpy(), color=labels[1][1])
    p.add_volume(volume.cpu().numpy(), cmap='gray', mapper='gpu' if device.startswith('cuda') else None)
    p.add_legend(labels=labels, face='-')

    p.link_views()
    p.show()
    p.close()
    pv.close_all()


if __name__ == '__main__':
    for idx in range(1, 20 + 1):
        infer(idx)
