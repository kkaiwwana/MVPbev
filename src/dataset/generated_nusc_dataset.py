import torch
import cv2
import numpy as np
import pandas as pd

from pyquaternion import Quaternion
from .utils import render_bitpacked_semantics, collate_fn


class GeneratedNuscDS(torch.utils.data.Dataset):
    def __init__(self, config):
        self.base_root = config['base_root']
        self.csv_filename = config['csv_filename']

        self.csv_data = pd.read_csv(self.base_root + self.csv_filename)

        if not config['given_fixed_split']:
            # random sample if a fixed split is not given.
            ds_len = config['num_samples'] if len(self.csv_data) > config['num_samples'] else len(self.csv_data)
            self.csv_data = self.csv_data.sample(ds_len, ignore_index=True)

        self.origin_size = config['origin_size']
        self.target_size = config['target_size']
        self.mask_foreground = config['mask_foreground']
        self.cam_order = config['cam_order']

    @staticmethod
    def _estimate_homography(R, K):
        m = R.shape[0]
        homos = torch.zeros((m, m, 3, 3), device=R.device)
        for i in range(m):
            for j in range(m):
                homo_l = K[j] @ torch.inverse(R[j]) @ R[i] @ torch.inverse(K[i])
                homos[i, j] = homo_l

        return homos

    def get_homography(self, frame_data):
        R = []
        K = []
        for cam in self.cam_order:
            cam_params = frame_data[cam]['cam_params']
            K.append(torch.tensor(cam_params['camera_intrinsic'], dtype=torch.float64))
            R.append(torch.tensor(Quaternion(cam_params['rotation']).rotation_matrix, dtype=torch.float64))
        R = torch.stack(R, dim=0)
        K = torch.stack(K, dim=0)
        return GeneratedNuscDS._estimate_homography(R, K).to(torch.float32)

    @staticmethod
    def scale_homography(homos, origin_size, target_size):
        sc_x = target_size['w'] / origin_size['w']
        sc_y = target_size['h'] / origin_size['h']
        sc_mat = torch.tensor([[sc_x, 0, 0], [0, sc_y, 0], [0, 0, 1]], dtype=torch.float32)
        m = homos.shape[1]
        for i in range(m):
            for j in range(m):
                homos[i, j] = sc_mat @ homos[i, j] @ torch.inverse(sc_mat)

        return homos

    def __getitem__(self, idx):
        if not isinstance(idx, str):
            data_path = self.base_root + self.csv_data.loc[idx]['filepath']
        else:
            data_path = idx

        data = torch.load(data_path)

        homos = self.get_homography(data)
        homos = GeneratedNuscDS.scale_homography(homos, self.origin_size, self.target_size)

        images, foreground_masks, semantics, prompts = [], [], [], []
        _semantics = []

        for cam in self.cam_order:
            image = data[cam]['rgb_image']
            _boxes_mask = np.zeros((image.shape[0], image.shape[1])) != 0
            # get boxes & labels
            boxes = data[cam]['annotations']['boxes']
            labels = data[cam]['annotations']['labels']

            if boxes is not None:
                boxes *= (boxes >= 0)
                f_w = self.target_size['w'] / self.origin_size['w']
                f_h = self.target_size['h'] / self.origin_size['h']
                boxes = (boxes * [[f_w, f_h, f_w, f_h]]).astype(np.int64)

                for box, label in zip(boxes, labels):
                    if label.split('.')[0] not in ['vehicle']:
                        continue

                    if self.mask_foreground:
                        image[box[1]: box[3], box[0]: box[2]] = 0

                    _boxes_mask[box[1]: box[3], box[0]: box[2]] = True

            images.append(image)
            prompts.append(data[cam]['prompt'])

            seman_map = data[cam]['objects']['semantic_map'][:, :, None]
            # 13: car, color: [0, 102, 200];  14: truck, color: [255, 0, 21]; 15: bus, color: [253, 0, 241]
            car_mask = (seman_map == 13) & _boxes_mask[:, :, None]
            truck_mask = (seman_map == 14) & _boxes_mask[:, :, None]
            bus_mask = (seman_map == 15) & _boxes_mask[:, :, None]

            foreground_mask = car_mask | truck_mask | bus_mask
            foreground_masks.append(foreground_mask)

            drivable_seman = render_bitpacked_semantics(data[cam]['semantic'])
            _drivable_seman = render_bitpacked_semantics(data[cam]['semantic'], include_drivable=True)

            _zeros = np.zeros_like(drivable_seman)
            _ones = np.ones_like(drivable_seman)

            car_seman = (_zeros * ~car_mask + _ones * car_mask * np.array([0, 102, 200])[None, None])
            truck_seman = (_zeros * ~truck_mask + _ones * truck_mask * np.array([255, 0, 21])[None, None])
            bus_seman = (_zeros * ~bus_mask + _ones * bus_mask * np.array([253, 0, 241])[None, None])

            fg_seman = car_seman + truck_seman + bus_seman

            semantics.append(drivable_seman * ~foreground_mask + fg_seman * foreground_mask)
            _semantics.append(_drivable_seman * ~foreground_mask + fg_seman * foreground_mask)

        for i in range(len(images)):
            images[i] = cv2.resize(
                images[i], dsize=(self.target_size['w'], self.target_size['h']), interpolation=cv2.INTER_AREA
            )
            if sum([(seman == 0).all().item() for seman in semantics]) >= 3:
                semantics = _semantics  # shadow version of semantic which include real 'drivable' area

            semantics[i] = cv2.resize(
                semantics[i].astype(np.uint8),
                dsize=(self.target_size['w'], self.target_size['h']),
                interpolation=cv2.INTER_AREA,
            )
            foreground_masks[i] = cv2.resize(
                foreground_masks[i].astype(np.uint8),
                dsize=(self.target_size['w'] // 8, self.target_size['h'] // 8),
                interpolation=cv2.INTER_AREA,
            )

        return {
            'image_paths': data_path,
            'scene_id': data['meta']['scene_name'],
            'token': data['token'],
            'images': ((np.stack(images, axis=0) / 127.5) - 1).astype(np.float32),
            'prompt': prompts,
            'semantics': (((np.stack(semantics, axis=0) / 127.5) - 1).astype(np.float32)),
            'homos': homos,
            'foreground_mask': (np.stack(foreground_masks, axis=0).astype(bool)),
            'seman_control_mask': {
                _scale_id: np.stack(
                    [cv2.resize(seman, dsize=None, fx=1 / _scale, fy=1 / _scale) for seman in semantics]).sum(-1) != 0
                for _scale_id, _scale in zip(['raw', '1x', '2x', '4x', '8x'], [1, 8, 16, 32, 64])
            },
        }

    def __len__(self):
        return len(self.csv_data)

    def get_loader(self, batch_size, num_workers=0, shuffle=False, drop_last=False, **kwargs):

        dl = torch.utils.data.DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last,
            collate_fn=collate_fn,
            **kwargs
        )
        return dl

    def get_empty_data(self):
        h, w = self.target_size['h'], self.target_size['w']
        return {
            'image_paths': [''] * 6,
            'scene_id': '',
            'token': '',
            'images': np.zeros((6, h, w, 3), dtype=np.float32),
            'prompt': [''] * 6,
            'semantics': np.zeros((6, h, w, 3), dtype=np.float32),
            'homos': torch.tensor(np.eye(3)[None, None].repeat(6, 0).repeat(6, 1), dtype=torch.float32),
            'foreground_mask': np.zeros((6, h // 8, w // 8)) == 0,
            'seman_control_mask': {
                _scale_id: np.zeros((6, h // _scale, w // _scale), dtype=np.float32)
                for _scale_id, _scale in zip(['raw', '1x', '2x', '4x', '8x'], [1, 8, 16, 32, 64])
            },
        }