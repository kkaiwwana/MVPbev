import torch
import numpy as np
import torch.nn.functional as F
from einops import rearrange, repeat


MAX_COORD_X = 1600
CENTER_POINT_X = 448 // 2


def get_x_2d(width, height):
    x = np.arange(width)
    y = np.arange(height)
    x, y = np.meshgrid(x, y)
    z = np.ones_like(x)
    xyz = np.concatenate([x[..., None], y[..., None], z[..., None]], axis=-1).astype(np.float32)
    return xyz


def get_correspondences(homos, img_h, img_w, device):
    # homos with shape (batch, m, m, 3, 3)
    m = homos.shape[1]
    dtype = homos.dtype
    correspondences = torch.zeros((homos.shape[0], m, m, img_h, img_w, 2), device=device, dtype=dtype)
    # correspondence[:, i, j] = transformed coordinates in CAM_i for pixels in CAM_j
    for i in range(m):
        for j in range(m):
            # m = multi-view num
            # correspondences -> (batch, m, m, h, w, 2)
            
            # homo_l -> (1, 3, 3)
            # left to right, i to j
            homo_l = homos[:, i, j]
            
            # xyz_l, coordinates -> (h, w, 3), i.e. h x w x [x, y, 1]
            xyz_l = torch.tensor(get_x_2d(img_w, img_h), device=device, dtype=dtype)
            xyz_l = (xyz_l.reshape(-1, 3))[None].repeat(homo_l.shape[0], 1, 1)
            
            # transformation, (1, 3, 3) @ (1, 3, 512x512)
            xyz_l = (xyz_l @ homo_l.transpose(-1, -2)).transpose(-1, -2)
            
            # unify z coordinates
            # xy_l -> (batch, 1, h, w, [x, y coords transformed])
            xy_l = torch.reshape(xyz_l[:, :2] / xyz_l[:, 2:], (-1, 1, 2, img_h, img_w)).permute(0, 1, -2, -1, -3)

            # set max coord 1600
            _mask = (xy_l - CENTER_POINT_X).abs() <= 2 * MAX_COORD_X
            xy_l = xy_l * _mask + 2 * MAX_COORD_X * ~_mask
            
            correspondences[:, i, j] = xy_l[:, 0]

    return correspondences


def get_key_value(key_value, xy_l, homo_r, ori_h, ori_w, ori_h_r, query_h, kernel_size=3):
    # 3x times boosted than original impl.
    b, c, h, w = key_value.shape
    query_scale = ori_h // query_h
    key_scale = ori_h_r // h

    xy_l = xy_l[:, query_scale // 2::query_scale, query_scale // 2::query_scale] / key_scale - 0.5

    offset = repeat(torch.zeros_like(xy_l), 'b ... -> b (l) ...', l=kernel_size ** 2).contiguous()
    xy_l = repeat(xy_l, 'b ... -> b (l) ...', l=kernel_size ** 2)

    for i, off_x in enumerate(range(0 - kernel_size // 2, 1 + kernel_size // 2)):
        for j, off_y in enumerate(range(0 - kernel_size // 2, 1 + kernel_size // 2)):
            offset[:, kernel_size * i + j, ..., 0] = off_x
            offset[:, kernel_size * i + j, ..., 1] = off_y

    xy_l = xy_l + offset
    xy_proj = (xy_l + 0.5) * key_scale

    xy_l[..., 0] = xy_l[..., 0] / (w - 1) * 2 - 1
    xy_l[..., 1] = xy_l[..., 1] / (h - 1) * 2 - 1

    key_values = F.grid_sample(
        input=repeat(key_value, 'b ... -> (b l) ...', l=kernel_size ** 2),
        grid=rearrange(xy_l, 'b l ... -> (b l) ...'),
        align_corners=True
    )
    key_values = rearrange(key_values, '(b l) ... -> b l ...', l=kernel_size ** 2)

    mask = (xy_proj[..., 0] > 0) * (xy_proj[..., 0] < ori_w) * (xy_proj[..., 1] > 0) * (xy_proj[..., 1] < ori_h)

    xy_proj_back = torch.cat([xy_proj, torch.ones(*xy_proj.shape[:-1], 1, device=xy_proj.device)], dim=-1)
    xy_proj_back = rearrange(xy_proj_back, 'b n h w c -> b c (n h w)')
    xy_proj_back = homo_r @ xy_proj_back

    xy_proj_back = rearrange(xy_proj_back, 'b c (n h w) -> b n h w c', h=h, w=w)
    xy_proj_back = xy_proj_back[..., :2] / xy_proj_back[..., 2:]

    xy = get_x_2d(ori_w, ori_h)[:, :, :2]
    xy = xy[query_scale // 2::query_scale, query_scale // 2::query_scale]
    xy = torch.tensor(xy, device=key_value.device).float()[None, None]

    xy_rel = (xy_proj_back - xy) / query_scale

    return key_values, xy_rel, mask


def get_query_value(query, key_value, xy_l, homo_r, img_h_l, img_w_l, img_h_r=None, img_w_r=None):
    if img_h_r is None:
        img_h_r = img_h_l
        img_w_r = img_w_l

    m = key_value.shape[1]  # m = 2, 2 neighboring views

    key_values, masks, xys = [], [], []

    for i in range(m):
        _, _, q_h, q_w = query.shape
        _key_value, _xy, _mask = get_key_value(
            key_value[:, i], xy_l[:, i], homo_r[:, i], img_h_l, img_w_l, img_w_r, q_h
        )

        key_values.append(_key_value)
        xys.append(_xy)
        masks.append(_mask)

    key_value = torch.cat(key_values, dim=1)
    xy = torch.cat(xys, dim=1)
    mask = torch.cat(masks, dim=1)

    return query, key_value, xy, mask
