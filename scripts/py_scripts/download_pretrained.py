import os
import argparse
import torch


MODELS = {
    'stable_diffusion_v15': 'runwayml/stable-diffusion-v1-5',
    'controlnet_v11_seg': 'lllyasviel/sd-controlnet-seg',
    'blip': 'Salesforce/blip-image-captioning-base',
    'mask2former': 'facebook/mask2former-swin-large-cityscapes-semantic'
    
}


def download_stable_diffusion_v15(path_to_save_model: str) -> None:
    from diffusers import AutoencoderKL, DDIMScheduler, UNet2DConditionModel
    from transformers import CLIPTextModel, CLIPTokenizer
    
    sd_model = MODELS['stable_diffusion_v15']
    
    tokenizer = CLIPTokenizer.from_pretrained(sd_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(sd_model, subfolder="text_encoder", torch_dtype=torch.float16)
    vae = AutoencoderKL.from_pretrained(sd_model, subfolder="vae")
    scheduler = DDIMScheduler.from_pretrained(sd_model, subfolder="scheduler")
    unet = UNet2DConditionModel.from_pretrained(sd_model, subfolder="unet")
    
    baseroot = os.path.join(path_to_save_model, 'stable_diffusion_v15/')
    os.makedirs(baseroot, exist_ok=True)
    tokenizer.save_pretrained(baseroot + 'tokenizer/')
    print(f'-- [{"CLIP_Tokenizer":^{20}}] --> Done.')
    text_encoder.save_pretrained(baseroot + 'text_encoder/')
    print(f'-- [{"CLIP_Text_Encoder":^{20}}] --> Done.')
    vae.save_pretrained(baseroot + 'vae/')
    print(f'-- [{"VAE":^{20}}] --> Done.')
    scheduler.save_pretrained(baseroot + 'scheduler/')
    print(f'-- [{"DDIM_Scheduler":^{20}}] --> Done.')
    unet.save_pretrained(baseroot + 'unet/')
    print(f'-- [{"UNet2D":^{20}}] --> Done.')
    
    
    
def download_controlnet_v11_seg(path_to_save_model: str):
    from diffusers import ControlNetModel
    ctrl_model = MODELS['controlnet_v11_seg']
    
    controlnet = ControlNetModel.from_pretrained(ctrl_model)
    
    baseroot = os.path.join(path_to_save_model, 'controlnet_v11_seg/')
    os.makedirs(baseroot, exist_ok=True)
    controlnet.save_pretrained(baseroot)
    print(f'-- [{"Controlnet":^{20}}] --> Done.')
    

parser = argparse.ArgumentParser(prog='download_pretrained_mdoel')
parser.add_argument('--path_to_save_model', '--p', type=str, default='../../weights/pretrained/')
        

if __name__ == '__main__':
    arguments = parser.parse_args()
    p = arguments.path_to_save_model
    print('Start Downloading:')
    download_controlnet_v11_seg(p)
    download_stable_diffusion_v15(p)
    