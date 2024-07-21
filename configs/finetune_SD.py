cam_order = [
    'CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 
    'CAM_BACK_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT'
]

dataset = {
    'name': 'GeneratedNuscDS',
    
    'train': {
        'cam_order': cam_order,
        'base_root': '/root/autodl-tmp/GeneratedNusc/',
        'csv_filename': 'train.csv',
        'given_fixed_split': False,  # will not use num_samples when a fixed splits given (used for test)
        'num_samples': 6000,
        'origin_size': {
            'w': 1600,
            'h': 900,
        },
        'target_size': {
            'w': 448,
            'h': 256,
        },
        'mask_foreground': False,
    },
    'valid': {
        'cam_order': cam_order,
        'base_root': '/root/autodl-tmp/GeneratedNusc/',
        'csv_filename': 'valid.csv',
        'given_fixed_split': False,
        'num_samples': 9999,  # valid set will not be fully used while training. it's time-consuming.
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
        'requires_grad': True,

        'guidance_scale': 7.5,
        'diff_timestep': 50,
    },
    'Controlnet': {
        'Ctrl_path': 'lllyasviel/sd-controlnet-seg',
        'Ctrl_path_local': '/root/autodl-tmp/MVPbev/weights/pretrained/controlnet_v11_seg',
        'from_local': False,
        'requires_grad': False,

        'add_control_mask': True,  # mask control signal outside the semantic area
        
        'train_mode': False  # training from scratch
    },
    'MVAttn': {
        'is_enabled': False,
        'requires_grad': False,
    },
    'others': {
        'enforce_cross_view_consistency': False,
        'enforce_cross_view_consistency_steps': 30,
    }
}

# support 1-gpu so far.
train = {
    'epoch': 16,
    'batch_size': 6,
    'num_workers': 16,
    'device': 'cuda',
    'optimizer': {
        'name': 'AdamW',
        'lr': 5e-7,
        'weight_decay': 1e-5,
        'momentum': None,
    },
    'lr_scheduler': {
        'name': 'CosineAnnealingLR',
    },
    'grad_accu_steps': 3,  # gradient accumulation, >1: enable
    
    'log_every_n_steps': 50,
    'sanity_check_val': 1,
    'sanity_check_train': 1,
    'limit_val_batches': 4,
    'ckpt_path': None,
}