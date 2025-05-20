import torch
import torch.nn.functional as F


def get_axis(series, eps=1e-20):
    p1p2 = series[:, 1:3, :] - series[:, 0:1, :]
    ax_x = p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_x = F.normalize(ax_x, p=2.0, dim=-1, eps=eps)
    ax_y = -p1p2[:, 1, :] - p1p2[:, 0, :]
    ax_y = F.normalize(ax_y, p=2.0, dim=-1, eps=eps)
    ax_z = torch.cross(ax_x, ax_y, dim=-1)
    ax_z = F.normalize(ax_z, p=2.0, dim=-1, eps=eps)
    axis = torch.stack([ax_x, ax_y, ax_z], dim=1)
    return axis


def euler_from_matrix(matrix, eps=1e-6):
    i, j, k = 0, 1, 2
    M = matrix[:, :3, :3]

    cy = torch.sqrt(M[:, i, i] * M[:, i, i] + M[:, j, i] * M[:, j, i])
    ax = torch.atan2(M[:, k, j], M[:, k, k])
    ay = torch.atan2(-M[:, k, i], cy)
    az = torch.atan2(M[:, j, i], M[:, i, i])
    flag = cy <= eps
    ax2 = torch.atan2(-M[:, j, k], M[:, j, j])
    ax[flag, ...] = ax2[flag, ...]
    az[flag, ...] = 0

    a = torch.stack([ax, ay, az], dim=-1)
    return a


def euler_matrix(angle):
    i, j, k = 0, 1, 2
    ai, aj, ak = angle[:, 0], angle[:, 1], angle[:, 2]

    si, sj, sk = torch.sin(ai), torch.sin(aj), torch.sin(ak)
    ci, cj, ck = torch.cos(ai), torch.cos(aj), torch.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    M = torch.eye(4, dtype=ai.dtype, device=ai.device).unsqueeze(0).repeat(len(ai), 1, 1)
    M[:, i, i] = cj * ck
    M[:, i, j] = sj * sc - cs
    M[:, i, k] = sj * cc + ss
    M[:, j, i] = cj * sk
    M[:, j, j] = sj * ss + cc
    M[:, j, k] = sj * cs - sc
    M[:, k, i] = -sj
    M[:, k, j] = cj * si
    M[:, k, k] = cj * ci

    return M


def affine_matrix_from_points(v0, v1):
    t0 = -torch.mean(v0, dim=-1)
    v0 = v0 + t0.unsqueeze(-1)
    t1 = -torch.mean(v1, dim=-1)
    v1 = v1 + t1.unsqueeze(-1)

    u, s, vh = torch.svd(torch.bmm(v1, v0.permute(0, 2, 1)).cpu())
    if u.device != v0.device:
        u, vh = torch.cat([u, vh], dim=-1).to(v0.device).split(3, dim=-1)
    vh = vh.permute(0, 2, 1)
    R = torch.bmm(u, vh)

    flag = torch.det(R) < 0.0
    out = u[:, :, 2:3] * (vh[:, 2:3, :] * 2.0)
    R[flag, ...] = R[flag, ...] - out[flag, ...]

    M = torch.cat([R, torch.sum(R * t0.unsqueeze(1), dim=-1, keepdim=True) - t1.unsqueeze(-1)], dim=-1)
    M = F.pad(M, [0, 0, 0, 1])
    M[:, -1, -1] = 1.0
    return M


def series_to_dof(series):
    angle_mat = get_axis(series[:-1]).permute(0, 2, 1)
    angle_mat_inv = torch.inverse(angle_mat)

    p0p1 = torch.bmm(torch.cat([angle_mat_inv, angle_mat_inv], dim=0), torch.cat([series[:-1, :, :] - series[:-1, 0:1, :], series[1:, :, :] - series[:-1, 0:1, :]], dim=0).permute(0, 2, 1))
    trmat_ax_p0 = affine_matrix_from_points(p0p1[:len(angle_mat_inv)], p0p1[len(angle_mat_inv):])
    angle_ax_p0 = euler_from_matrix(trmat_ax_p0)

    dist_ax_p0_tr = trmat_ax_p0[:, :3, 3]

    dofs = torch.cat([dist_ax_p0_tr, angle_ax_p0], dim=-1)
    return dofs


def dof_to_series(start_point, dof):
    b, t, _ = dof.shape
    dof = dof.view(b * t, -1)
    matrix = euler_matrix(dof[:, 3:])
    matrix[:, :3, 3] = dof[:, :3]
    matrix = matrix.view(b, t, 4, 4)

    start_axis = get_axis(start_point).permute(0, 2, 1)
    start_matrix = torch.cat([start_axis, start_point[:, 0, :].unsqueeze(-1)], dim=-1)
    start_matrix = F.pad(start_matrix, (0, 0, 0, 1))
    start_matrix[:, 3, 3] = 1
    start_matrix_inv = torch.inverse(start_matrix)

    matrix_chain = [start_matrix]
    for idx in range(matrix.shape[1]):
        matrix_chain.append(torch.bmm(matrix_chain[-1], matrix[:, idx]))
    matrix_chain = torch.stack(matrix_chain, dim=1)

    start_point_4d = F.pad(start_point, (0, 1))
    start_point_4d[:, :, 3] = 1
    series = torch.einsum('btij,bjk,bkl->btil', matrix_chain, start_matrix_inv, start_point_4d.permute(0, 2, 1)).permute(0, 1, 3, 2)[..., :3]

    return series
