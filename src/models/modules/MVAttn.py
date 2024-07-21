import torch
import torch.nn as nn
from einops import rearrange
from .base_modules.transformer import BasicTransformerBlock, PosEmbedding
from .utils import get_query_value


class MVAttn(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.transformer = BasicTransformerBlock(dim, dim // 32, 32, context_dim=dim)
        self.pe = PosEmbedding(2, dim // 4)

    def forward(self, x, correspondences, img_h, img_w, homos, m):
        b, c, h, w = x.shape
        x = rearrange(x, '(b m) c h w -> b m c h w', m=m)
        outs = []

        for i in range(m):
            indices = [(i - 1 + m) % m, (i + 1) % m]
            # coordinate relationship
            xy_l = correspondences[:, i, indices]
            # target view
            x_left = x[:, i]
            # neighbor view, for view 0 the neighbor view is 1 and 7
            x_right = x[:, indices]  # (batch, 2, C, H, W), H, W are down-sampled (e.g. h, w // 8), C is 320

            # homo_r -> (1, 2, 3, 3)
            # right to left, right means neighbors
            homo_r = homos[:, indices, i]

            # query -> (batch, C, H, W)
            # key_value -> (batch, 18, C, H, W)
            # key_value_xy -> (batch, l=18, H, W, C=2)
            # mask -> (batch, 18, 64, 64)
            query, key_value, key_value_xy, mask = get_query_value(x_left, x_right, xy_l, homo_r, img_h, img_w)
            # (b x h x w = 4096, l, c=2)
            key_value_xy = rearrange(key_value_xy, 'b l h w c -> (b h w) l c')
            # (bhw, l, c=320)
            key_value_pe = self.pe(key_value_xy)

            # (4096, l=18, C=320)
            key_value = rearrange(key_value, 'b l c h w -> (b h w) l c')
            mask = rearrange(mask, 'b l h w -> (b h w) l')

            key_value = (key_value + key_value_pe) * mask[..., None]

            query = rearrange(query, 'b c h w -> (b h w) c')[:, None]
            query_pe = self.pe(torch.zeros(query.shape[0], 1, 2, device=query.device))

            out = self.transformer(query, key_value, query_pe=query_pe)

            out = rearrange(out[:, 0], '(b h w) c -> b c h w', h=h, w=w)
            outs.append(out)
        out = torch.stack(outs, dim=1)

        # (8, 320, 64, 64) save as x (1, 8, 320, 64, 64)
        out = rearrange(out, 'b m c h w -> (b m) c h w')

        return out

