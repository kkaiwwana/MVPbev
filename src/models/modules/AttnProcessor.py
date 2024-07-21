import torch
import torch.nn.functional as F
from diffusers.models.attention_processor import Attention, AttnProcessor
from einops import rearrange


def build_max_pooling_layers(ori_size: tuple, tar_size: tuple):
    layers = []
    # h 2x downsample
    down_2x_num_h = int(torch.log2(ori_size[0] // tar_size[0]).floor().item())
    layers += [torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(2, 1)) for i in range(down_2x_num_h)]
    h = ori_size[0] // 2 ** down_2x_num_h
    # w 2x downsample
    down_2x_num_w = int(torch.log2(ori_size[1] // tar_size[1]).floor().item())
    layers += [torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 2)) for i in range(down_2x_num_w)]
    w = ori_size[1] // 2 ** down_2x_num_w

    # h-wise pool
    h_wise_pool_num = (h - tar_size[0]) // 2
    layers += [torch.nn.MaxPool2d(kernel_size=(3, 1), stride=(1, 1)) for i in range(h_wise_pool_num)]
    h -= 2 * h_wise_pool_num
    # w-wise pool
    w_wise_pool_num = (w - tar_size[1]) // 2
    layers += [torch.nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 1)) for i in range(w_wise_pool_num)]
    w -= 2 * w_wise_pool_num
    # final-h
    final_h_num = h - tar_size[0]
    layers += [torch.nn.MaxPool2d(kernel_size=(2, 1), stride=(1, 1)) for i in range(final_h_num)]
    # final-w
    final_w_num = w - tar_size[1]
    layers += [torch.nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)) for i in range(final_w_num)]

    return torch.nn.Sequential(*layers)


def mask_centered_max_pool(x_in, mask):
    # mask -> (B, num-obj, h, w)
    # x_in -> (B, o, c, h, w)
    _, _, _, h, w = x_in.shape

    for b_i in range(mask.shape[0]):
        for o_i in range(mask.shape[1]):
            h_idx, w_idx = torch.where(mask[b_i, o_i])  # one view

            if len(h_idx) == 0:
                continue
            else:
                h_0, h_1 = min(h_idx), max(h_idx)
                w_0, w_1 = min(w_idx), max(w_idx)

                max_pool_layers = build_max_pooling_layers((h, w), (h_1 - h_0 + 1, w_1 - w_0 + 1))
                x = max_pool_layers(x_in[b_i, o_i])  # c h w
                x_in[b_i, o_i, :, h_0: h_1 + 1, w_0: w_1 + 1] = x

    return x_in


class CrossAttnProcessor(AttnProcessor):
    def __init__(self, attn_masks, is_cfg):
        self.attn_masks = attn_masks
        self.is_cfg = is_cfg  # cfg stands for Classifier-Free-Guidance

    def __call__(self, attn: Attention, hidden_states, encoder_hidden_states=None, **kwargs):

        # is_cross-attn
        if encoder_hidden_states is not None:

            bg_hidden_states, *fg_hidden_states = encoder_hidden_states.chunk(chunks=5, dim=-2)

            fg_hidden_states = torch.stack(fg_hidden_states, dim=1)  # (B, 4-obj, seq_len, hid_dim)
            fg_hidden_states = rearrange(fg_hidden_states, 'b o l d -> (o b) l d')

            bg_res = super().__call__(attn, hidden_states, bg_hidden_states, **kwargs)  # (b * m, 32* 56, 320)
            fg_res = super().__call__(attn, hidden_states.repeat(4, 1, 1), fg_hidden_states,
                                      **kwargs)  # (b * m * o, 32* 56, 320)
            fg_res = rearrange(fg_res, '(o b) hw c -> b hw c o', o=4)

            mask = rearrange(self.attn_masks, 'b m h w o -> (b m) o h w')  # (0, 1) mask
            sc_factor = (bg_res.shape[1] / (mask.shape[-1] * mask.shape[-2])) ** (1 / 2)
            mask = F.interpolate(mask, scale_factor=sc_factor, mode='bilinear')

            # _, _, h, w = mask.shape
            # fg_res = rearrange(fg_res, 'b (h w) c o -> b o c h w ', h=h, w=w)
            # fg_res = mask_centered_max_pool(fg_res, mask)
            # fg_res = rearrange(fg_res, 'b o c h w -> b (h w) c o')

            mask = rearrange(mask, 'B o h w -> B (h w) o')

            if self.is_cfg:
                mask = mask.repeat(2, 1, 1)  # support CFG

            return sum([fg_res[..., i] * mask[..., i: i + 1] for i in range(mask.shape[-1])]) + bg_res * (
                        mask.sum(dim=-1, keepdim=True) == 0)

        # is self-attn
        else:
            return super().__call__(attn, hidden_states, encoder_hidden_states, **kwargs)
