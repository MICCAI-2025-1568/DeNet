import torch
import torch.nn.functional as F


def get_reco_size(series):
    series = torch.cat([series, 2 * series[:, 0:1, :] - series[:, 1:2, :], 2 * series[:, 0:1, :] - series[:, 2:3, :]], dim=1)
    min_point = torch.min(series.view(-1, 3), dim=0)[0]
    max_point = torch.max(series.view(-1, 3), dim=0)[0]
    range_point = max_point - min_point + 1
    reco_size = torch.ceil(range_point).int()
    bias = min_point - 0.5
    return reco_size, bias


def get_axis(series, eps: float = 1e-20):
    p1p2 = series[:, 1:3, :] - series[:, 0:1, :]
    ax_x = p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_x = F.normalize(ax_x, p=2.0, dim=-1, eps=eps)
    ax_y = -p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_y = F.normalize(ax_y, p=2.0, dim=-1, eps=eps)
    ax_z = torch.cross(ax_x, ax_y, dim=-1)
    ax_z = F.normalize(ax_z, p=2.0, dim=-1, eps=eps)
    axis = torch.stack([ax_x, ax_y, ax_z], dim=1)
    return axis


def transform(points, height, width):
    axis = get_axis(points).permute(0, 2, 1)
    range_x = torch.arange(-(height - 1) / 2, (height + 1) / 2, dtype=points.dtype, device=points.device)
    range_y = torch.arange(-(width - 1) / 2, (width + 1) / 2, dtype=points.dtype, device=points.device)
    mesh_x, mesh_y = torch.meshgrid(range_x, range_y, indexing='ij')
    mesh = torch.stack([mesh_y, -mesh_x, torch.zeros_like(mesh_x)], dim=-1)
    center = points[:, 0, :].unsqueeze(1).unsqueeze(1)
    local_mesh = torch.einsum('nij,hwj->nhwi', axis, mesh) + center
    return local_mesh


def reco(source, series):
    reco_size, bias = get_reco_size(series)
    volume = torch.zeros(*reco_size, dtype=source.dtype, device=source.device)
    series = series - bias
    count = torch.zeros(*reco_size, dtype=torch.int, device=source.device)
    for idx in range(len(source)):
        mesh = transform(series[idx:idx + 1], *source.shape[-2:])
        tib = mesh.int().reshape(-1, 3)
        tib0, tib1, tib2 = tib.split(1, dim=-1)
        flag = (tib0 < 0) | (tib0 >= reco_size[0]) | (tib1 < 0) | (tib1 >= reco_size[1]) | (tib2 < 0) | (tib2 >= reco_size[2])
        flag = ~flag
        volume[tib0[flag], tib1[flag], tib2[flag]] += source[idx].view(-1, 1)[flag].type(volume.dtype)
        count[tib0[flag], tib1[flag], tib2[flag]] += 1
    count = torch.where(count > 0, count, 1)
    volume = volume / count
    return volume, bias
