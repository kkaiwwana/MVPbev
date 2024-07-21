cam_order = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]

dataset = {
    'name': 'GeneratedNuscDS',
    'valid': {
        'cam_order': cam_order,
        'base_root': '/root/autodl-tmp/GeneratedNusc/',
        'csv_filename': 'valid.csv',
        'given_fixed_split': False,  # will not use num_samples when a fixed splits given (used for test)
        'num_samples': 1200,
        'origin_size': {
            'w': 1600,
            'h': 900,
        },
        'target_size': {
            'w': 448,
            'h': 256,
        },
        'mask_foreground': False,
    }
}

model = {
    'name': 'MultiViewLDM',
    'SD': {
        'SD_path': 'runwayml/stable-diffusion-v1-5',  # path in Huggingface.co
        'SD_path_local': 'root/autodl-tmp/MVPbev/weights/pretrained/stable_diffusion_v15',
        'from_local': False,
        'requires_grad': False,

        'guidance_scale': 7.5,
        'diff_timestep': 50,
    },
    'Controlnet': {
        'Ctrl_path': 'lllyasviel/sd-controlnet-seg',
        'Ctrl_path_local': '/root/autodl-tmp/MVPbev/weights/pretrained/controlnet_v11_seg',
        'from_local': False,
        'requires_grad': False,

        'add_control_mask': True,  # mask control info outside the semantic area

        'train_mode': False  # train from scratch
    },
    'MVAttn': {
        'is_enabled': True,
        'requires_grad': False,
    },
    'others': {
        'enforce_cross_view_consistency': True,
        'enforce_cross_view_consistency_steps': 35,
    }
}

# support 1-gpu so far.
test = {
    'batch_size': 8,
    'num_workers': 16,
    'device': 'cuda',
    'ckpt_path': '/root/autodl-tmp/MVPbev/weights/checkpoints/2024_4_10_ALL_ON.ckpt'
}