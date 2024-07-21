import torch
import numpy as np


def render_bitpacked_semantics(semantic, classes=6, include_drivable=False):
    # n_colors corresponded to n_classes in create_dataset config
    n_colors = [
        [140, 140, 140] if include_drivable else [0, 0, 0],
        [140, 140, 140],
        [140, 140, 140],
        [0, 0, 0],
        [0, 0, 0],
    ]
    semantics = np.unpackbits(semantic, axis=-1)[:, :, :classes]

    h, w = semantic.shape[:2]

    semantic_masks = []
    for seman_i, color in zip(range(semantics.shape[-1]), n_colors):
        semantic = semantics[:, :, seman_i]
        pure_color_mask = np.ndarray((h, w, 3), dtype=np.uint8)
        pure_color_mask[:, :, :] = np.array(color)
        pure_color_mask *= semantic.reshape(h, w, 1)
        semantic_masks.append(pure_color_mask)

    masks = np.stack([mask.sum(axis=-1, keepdims=True) != 0 for mask in semantic_masks], axis=0).astype(np.uint8)

    result = np.ndarray((h, w, 3), dtype=np.uint8)
    result[:, :, :] = np.array([140, 140, 140])

    result = result * (masks.any(axis=0))

    return result


def collate_fn(datas):
    images = torch.tensor(np.stack([data['images'] for data in datas], axis=0))
    semantics = torch.tensor(np.stack([data['semantics'] for data in datas], axis=0))
    prompt = [data['prompt'] for data in datas]
    homos = torch.stack([data['homos'] for data in datas], dim=0)
    image_paths = [data['image_paths'] for data in datas]
    scene_id = [data['scene_id'] for data in datas]
    token = [data['token'] for data in datas]
    
    fore_mask = torch.tensor(np.stack([data['foreground_mask'][:, None, :, :] for data in datas], axis=0))
    control_mask = {
        'raw': torch.tensor(np.stack([data['seman_control_mask']['raw'][:, None] for data in datas], axis=0)),
        '1x': torch.tensor(np.stack([data['seman_control_mask']['1x'][:, None] for data in datas], axis=0)),
        '2x': torch.tensor(np.stack([data['seman_control_mask']['2x'][:, None] for data in datas], axis=0)),
        '4x': torch.tensor(np.stack([data['seman_control_mask']['4x'][:, None] for data in datas], axis=0)),
        '8x': torch.tensor(np.stack([data['seman_control_mask']['8x'][:, None] for data in datas], axis=0)),
    }

    return {
        'images': images,
        'semantics': semantics,
        'prompt': prompt,
        'homos': homos,
        'image_paths': image_paths,
        'scene_id': scene_id,
        'token': token,
        'foreground_mask': fore_mask,
        'seman_control_mask': control_mask
    }