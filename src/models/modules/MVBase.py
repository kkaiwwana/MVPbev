import torch.nn as nn
from .MVAttn import MVAttn
from einops import rearrange
from .utils import get_correspondences


class MultiViewBaseModel(nn.Module):
    def __init__(self, unet, enable_MVAttn=False):
        super().__init__()

        self.unet = unet
        self.enable_MVAttn = enable_MVAttn

        if not self.enable_MVAttn:
            self.trainable_parameters = [(self.unet.parameters(), 1.0)]
        else:
            self.cp_blocks_encoder = nn.ModuleList()
            for i in range(len(self.unet.down_blocks)):
                self.cp_blocks_encoder.append(MVAttn(self.unet.down_blocks[i].resnets[-1].out_channels))

            self.cp_blocks_mid = MVAttn(self.unet.mid_block.resnets[-1].out_channels)

            self.cp_blocks_decoder = nn.ModuleList()
            for i in range(len(self.unet.up_blocks)):
                self.cp_blocks_decoder.append(MVAttn(self.unet.up_blocks[i].resnets[-1].out_channels))

            self.trainable_parameters = [(
                list(self.cp_blocks_mid.parameters()) +
                list(self.cp_blocks_decoder.parameters()) +
                list(self.cp_blocks_encoder.parameters()),
                1.0
            )]

    def forward(self, latents, timestep, prompt_embd, meta, control=None):

        homos = meta['homos']

        b, m, c, h, w = latents.shape
        dtype = latents.dtype
        img_h, img_w = h * 8, w * 8

        correspondences = get_correspondences(homos, img_h, img_w, latents.device)

        # bs * m, 4, 64, 64
        sample = rearrange(latents, 'b m c h w -> (b m) c h w')
        encoder_hidden_states = rearrange(prompt_embd, 'b m l c -> (b m) l c')

        # 1. process timesteps
        timestep = timestep.reshape(-1)
        t_emb = self.unet.time_proj(timestep).to(dtype)  # (bs, 320)
        # print(t_emb.dtype)
        emb = self.unet.time_embedding(t_emb)  # (bs, 1280)

        sample = self.unet.conv_in(sample)  # bs*m, 320, 64, 6

        if control is not None:
            down_block_additional_residuals, mid_block_additional_residual = control[:2]
        else:
            down_block_additional_residuals, mid_block_additional_residual = None, None

        is_controlnet = mid_block_additional_residual is not None and down_block_additional_residuals is not None
        is_adapter = mid_block_additional_residual is None and down_block_additional_residuals is not None

        down_block_res_samples = (sample,)
        for i, downsample_block in enumerate(self.unet.down_blocks):
            if hasattr(downsample_block, "has_cross_attention") and downsample_block.has_cross_attention:
                additional_residuals = {}
                if is_adapter and len(down_block_additional_residuals) > 0:
                    additional_residuals["additional_residuals"] = down_block_additional_residuals.pop(0)

                sample, res_samples = downsample_block(
                    hidden_states=sample,
                    temb=emb,
                    encoder_hidden_states=encoder_hidden_states,
                    **additional_residuals,
                )
            else:
                sample, res_samples = downsample_block(hidden_states=sample, temb=emb)

                if is_adapter and len(down_block_additional_residuals) > 0:
                    sample += down_block_additional_residuals.pop(0)

            if m > 1 and self.enable_MVAttn:
                sample = self.cp_blocks_encoder[i](
                    sample, correspondences, img_h, img_w, homos, m)

            down_block_res_samples += res_samples

        if is_controlnet:
            new_down_block_res_samples = ()

            for down_block_res_sample, down_block_additional_residual in zip(
                    down_block_res_samples, down_block_additional_residuals
            ):
                down_block_res_sample = down_block_res_sample + down_block_additional_residual
                new_down_block_res_samples = new_down_block_res_samples + (down_block_res_sample,)

            down_block_res_samples = new_down_block_res_samples

        # 4. mid
        if self.unet.mid_block is not None:
            sample = self.unet.mid_block(
                sample,
                emb,
                encoder_hidden_states=encoder_hidden_states,
            )

            if m > 1 and self.enable_MVAttn:
                sample = self.cp_blocks_mid(sample, correspondences, img_h, img_w, homos, m)

            if (
                    is_adapter
                    and len(down_block_additional_residuals) > 0
                    and sample.shape == down_block_additional_residuals[0].shape
            ):
                sample += down_block_additional_residuals.pop(0)

        if is_controlnet:
            sample = sample + mid_block_additional_residual

        # 5. up
        for i, upsample_block in enumerate(self.unet.up_blocks):
            is_final_block = i == len(self.unet.up_blocks) - 1

            res_samples = down_block_res_samples[-len(upsample_block.resnets):]
            down_block_res_samples = down_block_res_samples[: -len(upsample_block.resnets)]

            # if we have not reached the final block and need to forward the
            # upsample size, we do it here
            if not is_final_block:
                upsample_size = down_block_res_samples[-1].shape[2:]
            else:
                upsample_size = None

            if hasattr(upsample_block, "has_cross_attention") and upsample_block.has_cross_attention:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    encoder_hidden_states=encoder_hidden_states,
                    upsample_size=upsample_size,
                )
            else:
                sample = upsample_block(
                    hidden_states=sample,
                    temb=emb,
                    res_hidden_states_tuple=res_samples,
                    upsample_size=upsample_size,
                )

            if m > 1 and self.enable_MVAttn:
                sample = self.cp_blocks_decoder[i](sample, correspondences, img_h, img_w, homos, m)

        # 6. post-process
        if self.unet.conv_norm_out:
            sample = self.unet.conv_norm_out(sample)
            sample = self.unet.conv_act(sample)
        sample = self.unet.conv_out(sample)
        sample = rearrange(sample, '(b m) c h w -> b m c h w', m=m)
        return sample
