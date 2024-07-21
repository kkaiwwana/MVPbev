import cv2
import diffusers
import torch
import numpy as np

from typing import *
from einops import rearrange
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel, ControlNetModel

from .modules.MVBase import MultiViewBaseModel
from .modules.AttnProcessor import CrossAttnProcessor


class MultiViewLDM(torch.nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # models
        self.diff_timestep = model_config['SD']['diff_timestep']
        self.guidance_scale = model_config['SD']['guidance_scale']

        self.tokenizer, self.text_encoder, self.vae, self.scheduler, unet = self.load_sd_model(model_config)
        self.controlnet = self.load_ctrl_model(model_config, unet if model_config['Controlnet']['train_mode'] else None)
        self.add_control_mask = model_config['Controlnet']['add_control_mask']
        self.mv_base_model = MultiViewBaseModel(unet, model_config['MVAttn']['is_enabled'])

        # weights
        for param in self.mv_base_model.unet.parameters():
            param.requires_grad = model_config['SD']['requires_grad']
        if self.controlnet is not None:
            for param in self.controlnet.parameters():
                param.requires_grad = model_config['Controlnet']['requires_grad']

        self.trainable_parameters = []

        if model_config['SD']['requires_grad'] or model_config['MVAttn']['is_enabled']:
            self.trainable_parameters.append(self.mv_base_model.trainable_parameters[0])
        if self.controlnet is not None and model_config['Controlnet']['requires_grad']:
            self.trainable_parameters.append((self.controlnet.parameters(), 1.0))

        # others
        self.enforce_overlaps = model_config['others']['enforce_cross_view_consistency']
        self.enforce_overlap_steps = model_config['others']['enforce_cross_view_consistency_steps']

    @staticmethod
    def load_sd_model(model_config):
        sd_config = model_config['SD']
        sd_model = sd_config['SD_path_local'] if sd_config['from_local'] else sd_config['SD_path']

        tokenizer = CLIPTokenizer.from_pretrained(
            sd_model, subfolder="tokenizer", torch_dtype=torch.float16, local_files_only=sd_config['from_local'])
        text_encoder = CLIPTextModel.from_pretrained(
            sd_model, subfolder="text_encoder", torch_dtype=torch.float16, local_files_only=sd_config['from_local'])
        vae = AutoencoderKL.from_pretrained(
            sd_model, subfolder="vae", local_files_only=sd_config['from_local'])
        vae.eval()
        scheduler = DDIMScheduler.from_pretrained(
            sd_model, subfolder="scheduler", local_files_only=sd_config['from_local'])
        unet = UNet2DConditionModel.from_pretrained(
            sd_model, subfolder="unet", local_files_only=sd_config['from_local'])

        return tokenizer, text_encoder, vae, scheduler, unet

    @staticmethod
    def load_ctrl_model(model_config, unet=None) -> (torch.nn.Module, bool):
        ctrl_config = model_config['Controlnet']
        ctrl_model = ctrl_config['Ctrl_path_local'] if ctrl_config['from_local'] else ctrl_config['Ctrl_path']

        if ctrl_model is None:
            return None

        if unet is not None:
            ctrl_net = ControlNetModel.from_unet(unet)
        else:
            ctrl_net = ControlNetModel.from_pretrained(ctrl_model)

        return ctrl_net

    @torch.no_grad()
    def encode_text(self, text, device):
        text_inputs = self.tokenizer(
            text, padding="max_length", max_length=self.tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        )
        text_input_ids = text_inputs.input_ids
        if hasattr(self.text_encoder.config, "use_attention_mask") and self.text_encoder.config.use_attention_mask:
            attention_mask = text_inputs.attention_mask.cuda()
        else:
            attention_mask = None
        prompt_embeds = self.text_encoder(
            text_input_ids.to(device), attention_mask=attention_mask)

        return prompt_embeds[0], prompt_embeds[1]

    @torch.no_grad()
    def encode_image(self, x_input, vae):
        b = x_input.shape[0]

        x_input = rearrange(x_input, 'b m h w c -> (b m) c h w')
        z = vae.encode(x_input).latent_dist  # (bs, 2, 4, 64, 64)

        z = z.sample()
        z = rearrange(z, '(b m) c h w -> b m c h w', b=b)

        # use the scaling factor from the vae config
        z = z * vae.config.scaling_factor
        return z

    @torch.no_grad()
    def decode_latent(self, latents, vae):
        b, m = latents.shape[0:2]
        latents = (1 / vae.config.scaling_factor * latents)
        images = []
        for j in range(m):
            image = vae.decode(latents[:, j]).sample
            images.append(image)
        image = torch.stack(images, dim=1)
        image = (image / 2 + 0.5).clamp(0, 1).cpu().float()
        image = rearrange(image, 'b m c h w -> b m h w c').numpy()
        image = (image * 255).round().astype('uint8')

        return image

    @staticmethod
    def register_attn_processor(network, batch, is_cfg=False, at_where: List[str] = None) -> None:
        if at_where is None:
            at_where = ['down', 'mid', 'up']
        cross_attn_processors = {}
        for k in network.attn_processors.keys():
            if k.split('_')[0] in at_where:
                cross_attn_processors[k] = CrossAttnProcessor(batch['prompted_objs']['obj_mask'], is_cfg)
            else:
                # keep default for other blocks
                cross_attn_processors[k] = network.attn_processors[k]

        network.set_attn_processor(cross_attn_processors)

    def training_step(self, batch: Dict) -> torch.Tensor:

        meta = {'homos': batch['homos']}
        device = batch['images'].device
        dtype = batch['images'].dtype
        bs, m, h, w, _ = batch['images'].shape

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(torch.concat([self.encode_text(p, device)[0] for p in prompt], dim=0))
            # (batch, m, 77, 768)

        prompt_embds = torch.stack(prompt_embds, dim=0).to(dtype)

        if 'prompted_objs' in batch.keys():
            prompted_objs_embds = []
            for obj_prompts in batch['prompted_objs']['obj_prompt']:
                prompted_objs_embds.append(
                    torch.concat([self.encode_text(prompt, device)[0] for prompt in obj_prompts], dim=0)  # m, 768
                )
            prompted_objs_embds = torch.stack(prompted_objs_embds, dim=0).to(dtype)  # b, m, 1, 768
            prompt_embds = torch.concat([prompt_embds, prompted_objs_embds], dim=-2)

        latents = self.encode_image(batch['images'], self.vae).to(dtype)

        t = torch.randint(0, self.scheduler.config.num_train_timesteps, (bs,), device=device).long()

        noise = torch.randn_like(latents)
        noise_z = self.scheduler.add_noise(latents, noise, t)
        t = t[:, None].repeat(1, m)

        if self.controlnet:
            control: diffusers.models.controlnet.ControlNetOutput = self.controlnet(
                rearrange(latents, 'b m c h w -> (b m ) c h w'),
                t.reshape(-1),
                rearrange(prompt_embds, 'b m h w -> (b m) h w'),
                rearrange(batch['semantics'], 'b m h w c -> (b m) c h w')
            )
            if self.add_control_mask and 'seman_control_mask' in batch.keys():
                control: List = self.add_seman_control_mask(control, batch['seman_control_mask'], h // 8, w // 8)
        else:
            control: object = None

        denoise = self.mv_base_model(noise_z, t, prompt_embds, meta, control)

        return ((noise - denoise) ** 2).mean()

    @staticmethod
    def gen_cls_free_guide_pair(latents, timestep, prompt_embd, batch, control):
        latents = torch.cat([latents] * 2)
        timestep = torch.cat([timestep] * 2)
        if control is not None:
            control = [[ct.repeat(2, 1, 1, 1) for ct in control[0]], control[1].repeat(2, 1, 1, 1)]

        homos = torch.cat([batch['homos']] * 2)
        meta = {'homos': homos}

        return latents, timestep, prompt_embd, meta, control

    @staticmethod
    def add_seman_control_mask(control: diffusers.models.controlnet.ControlNetOutput, mask, h, w):
        control = list(control[:2])
        scale_to_scale_id = {1: '1x', 2: '2x', 4: '4x', 8: '8x'}

        for i, cont in enumerate(control[0]):
            control[0][i] *= rearrange(mask[scale_to_scale_id[w // cont.shape[-1]]], 'b m c h w -> (b m) c h w')

        control[1] *= rearrange(mask[scale_to_scale_id[w // control[1].shape[-1]]], 'b m c h w -> (b m) c h w')

        return control

    @torch.no_grad()
    def forward_cls_free(self, latents_high_res, _timestep, prompt_embd, batch, model, control=None):
        latents, _timestep, _prompt_embd, meta, control = self.gen_cls_free_guide_pair(
            latents_high_res, _timestep, prompt_embd, batch, control)

        noise_pred = model(latents, _timestep, _prompt_embd, meta, control)
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)

        return noise_pred

    @torch.no_grad()
    def validation_step(self, batch):
        images_pred = self.inference(batch)
        images = ((batch['images'] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)
        return images_pred, images

    @torch.no_grad()
    def enforce_cross_view_consistency(self, latent: np.ndarray, homos: np.ndarray, device: str, dtype: torch.dtype):
        # I tried an implementation of warping pers-transformation in Kornia based on PyTorch.
        # I validate that with visualization, that looks fine. But somehow it makes results terrible.
        # So I keep original implementation here, which repeatedly moves tensors from GPU to RAM...and covert it
        # from Tensor to ndarray...that's bad. But this implementation doesn't make inference slower actually,
        # compared with Kornia impl where all operations are done in GPU. I think the extra cost here is ignorable
        # comparing to giant Diffusion model inference cost.
        #
        # for b_i in range(bs):
        #     overlap_r = warp_perspective(
        #         latents[b_i][[(i + 1) % m for i in range(m)]], # src: 1 2 3 4 5 0
        #         # 1 2 3 4 5 0 -> 0 1 2 3 4 5
        #         sc_mat @ homos[b_i][[(i + 1) % m for i in range(m)], [i for i in range(m)]] @ torch.inverse(sc_mat),
        #         dsize=(h // 8, w // 8)
        #     )
        #     mask_r = overlap_r.sum(dim=1, keepdims=True) != 0
        #     latents[b_i] = latents[b_i] * (~mask_r) + overlap_r
        #
        bs, m, h, w, _ = latent.shape

        views_latent = []
        sc_mat = np.array([[1 / 8, 0, 0], [0, 1 / 8, 0], [0, 0, 1]])
        for m_i in range(m):
            overlap_r = np.stack([
                cv2.warpPerspective(
                    latent[b_i, (m_i + 1) % m],
                    sc_mat @ homos[b_i, (m_i + 1) % m, m_i] @ np.linalg.inv(sc_mat),
                    dsize=(w, h),
                    flags=cv2.INTER_NEAREST
                ) for b_i in range(bs)], axis=0
            )
            mask_r = overlap_r.sum(-1, keepdims=True) != 0

            views_latent.append(latent[:, m_i] * (~mask_r) + overlap_r * mask_r)

        return rearrange(torch.tensor(np.stack(views_latent, axis=1)), 'b m h w c -> b m c h w').to(device).to(dtype)

    @torch.no_grad()
    def inference(self, batch, seed=None):
        images = batch['images']
        bs, m, H, W, _ = images.shape
        device = images.device
        dtype = images.dtype

        if seed is not None:
            torch.manual_seed(seed)

        latents = torch.randn(bs, m, 4, H // 8, W // 8, device=device, dtype=dtype)
        if self.enforce_overlaps:
            # consistent initialization
            homos = batch['homos'].cpu().numpy()
            non_overlaps = np.random.randn(bs, m, H // 8, W // 8, 4)
            latents = self.enforce_cross_view_consistency(non_overlaps, homos, device, dtype)

        prompt_embds = []
        for prompt in batch['prompt']:
            prompt_embds.append(torch.concat([self.encode_text(p, device)[0] for p in prompt], dim=0))

        prompt_embds = torch.stack(prompt_embds, dim=0).to(dtype)
        prompt_null = self.encode_text('', device)[0][:, None].repeat(bs, m, 1, 1).to(dtype)

        if 'prompted_objs' in batch.keys():
            prompted_objs = batch['prompted_objs']
            prompted_objs_embds = []
            n_objs = len(prompted_objs['obj_prompt'][0][0])  # 'obj_prompt' -> (b m n 1)
            for obj_prompts in prompted_objs['obj_prompt']:
                prompted_objs_embds.append(
                    torch.concat([torch.concat([self.encode_text(prompt, device)[0] for prompt in prompts], dim=-2)
                                  for prompts in obj_prompts
                                  ], dim=0)
                )
            prompted_objs_embds = torch.stack(prompted_objs_embds, dim=0).to(dtype)  # b, m, 77 * n, 768
            prompt_embds = torch.concat([prompt_embds, prompted_objs_embds], dim=-2)

            prompt_null_obj = self.encode_text('', device)[0][:, None].repeat(bs, m, n_objs, 1).to(dtype)
            prompt_null = torch.concat([prompt_null, prompt_null_obj], dim=-2)

            self.register_attn_processor(self.mv_base_model.unet, batch, is_cfg=True, at_where=['down', 'mid', 'up'])
            if self.controlnet is not None:
                self.register_attn_processor(self.controlnet, batch, is_cfg=False, at_where=['mid'])

        prompt_embd = torch.cat([prompt_null, prompt_embds])

        self.scheduler.set_timesteps(self.diff_timestep, device=device)
        timesteps = self.scheduler.timesteps

        for i, t in enumerate(timesteps):
            _timestep = torch.cat([t[None, None]] * m, dim=1)
            _timestep = _timestep.repeat(bs, 1)

            if self.controlnet:
                control = self.controlnet(
                    rearrange(latents, 'b m c h w -> (b m ) c h w'),
                    _timestep.reshape(-1),
                    rearrange(prompt_embds, 'b m h w -> (b m) h w'),
                    rearrange(batch['semantics'], 'b m h w c -> (b m) c h w')
                )
                if self.add_control_mask and 'seman_control_mask' in batch.keys():
                    control = self.add_seman_control_mask(control, batch['seman_control_mask'], H // 8, W // 8)
            else:
                control = None

            noise_pred = self.forward_cls_free(latents, _timestep, prompt_embd, batch, self.mv_base_model, control)

            latents = self.scheduler.step(noise_pred, t, latents).prev_sample

            if self.enforce_overlaps and i < self.enforce_overlap_steps:
                homos = batch['homos'].cpu().numpy()
                non_overlaps = rearrange(latents.cpu(), 'b m c h w -> b m h w c').numpy()
                latents = self.enforce_cross_view_consistency(non_overlaps, homos, device, dtype)

        images_pred = self.decode_latent(latents, self.vae)

        return images_pred

    @torch.no_grad()
    def test_step(self, batch):
        images_pred = self.inference(batch)
        images = ((batch['images'] / 2 + 0.5) * 255).cpu().numpy().astype(np.uint8)

        result = {
            'images_gt': images,
            'images_gen': images_pred,
            'homos': batch['homos'],
            'prompt': batch['prompt'],
            'raw_data_path': batch['image_paths'],
        }

        return result