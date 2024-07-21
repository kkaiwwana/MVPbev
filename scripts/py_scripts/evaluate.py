import torch
import argparse
import tqdm
import os
import sys
import cv2
import torchmetrics
import numpy as np
import pandas as pd
import json
import PIL.Image as Image
import torch.nn.functional as F

from pyquaternion import Quaternion
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.inception import InceptionScore
from torchmetrics.multimodal.clip_score import CLIPScore
from torchmetrics.image import PeakSignalNoiseRatio
from torchvision.transforms import CenterCrop
from einops import rearrange

REMOVE_MASK = True
ADD_MASK = False


class BEVIoU(torchmetrics.Metric):
    def __init__(self):
        super().__init__()
        self.add_state('bev_gt', default=[])
        self.add_state('bev_pred', default=[])

    def compute(self):
        IoUs = [(pred & gt).sum() / (pred | gt).sum() for (pred, gt) in zip(self.bev_pred, self.bev_gt)]

        return sum(IoUs) / len(IoUs)

    def update(self, pred, gt):
        self.bev_gt.append(gt)
        self.bev_pred.append(pred)


def get_extrinsic(Q, T):
    mats = []
    for q, t in zip(Q, T):
        mat = np.zeros((4, 4), dtype=np.float32)
        mat[:3, :3] = Quaternion(q).rotation_matrix
        mat[:3, 3] = t
        mat[3, 3] = 1
        mats.append(mat)
    return np.stack(mats)


def parse_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--exp_name', '--n', type=str)
    parser.add_argument('--output_dir', '--o', type=str, default='../../test_output/',
                        help='Directory where results are saved')

    # CVT
    parser.add_argument('--cvt_code_path', type=str,
                        default='../../../cross_view_transformers/',
                        help='code path of CVT(cross-view-transformer)')
    parser.add_argument('--cvt_ckpt_path', type=str,
                        default='../../../cross_view_transformers/model.ckpt',
                        help='checkpoint path of pretrained CVT model')

    return parser.parse_args()


class MVResultDataset(torch.utils.data.Dataset):
    def __init__(self, results_dir):
        self.base_root = results_dir
        self.targets = os.listdir(results_dir)

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return torch.load(os.path.join(self.base_root, self.targets[idx]))


if __name__ == '__main__':
    sys.path.append('../../')
    from src.models.modules.utils import get_correspondences

    torch.multiprocessing.set_start_method('spawn')

    args = parse_args()

    fid = FrechetInceptionDistance(feature=2048).cuda()
    fid_gt = FrechetInceptionDistance(feature=2048).cuda()
    inception = InceptionScore().cuda()
    inception_gt = InceptionScore().cuda()

    CS_MODEL_PATH = 'openai/clip-vit-base-patch16'
    cs = CLIPScore(model_name_or_path=CS_MODEL_PATH).cuda()
    cs_gt = CLIPScore(model_name_or_path=CS_MODEL_PATH).cuda()

    psnr = PeakSignalNoiseRatio(data_range=1.0).cuda()
    psnr_gt = PeakSignalNoiseRatio(data_range=1.0).cuda()
    psnrs, psnrs_gt = [], []

    # bev IoU
    bev_IoU = BEVIoU().cuda()
    sys.path.append(args.cvt_code_path)

    from cross_view_transformer.common import load_backbone

    cvt_model = load_backbone(args.cvt_ckpt_path).cuda()
    cvt_model.eval()
    cvt_cam_intrin = torch.tensor(
        [[[377.3588, 0.0000, 248.1723],
          [0.0000, 377.3588, 89.2746],
          [0.0000, 0.0000, 1.0000]],

         [[375.8439, 0.0000, 247.9764],
          [0.0000, 375.8439, 94.9954],
          [0.0000, 0.0000, 1.0000]],

         [[377.0246, 0.0000, 245.3366],
          [0.0000, 377.0246, 89.5863],
          [0.0000, 0.0000, 1.0000]],

         [[376.4958, 0.0000, 248.8731],
          [0.0000, 376.4958, 94.1504],
          [0.0000, 0.0000, 1.0000]],

         [[239.0673, 0.0000, 257.3332],
          [0.0000, 239.0673, 97.0655],
          [0.0000, 0.0000, 1.0000]],

         [[374.9889, 0.0000, 247.6131],
          [0.0000, 374.9889, 92.7645],
          [0.0000, 0.0000, 1.0000]]]
    )


    def post_process_cvt_pred(pred, threshold=0.4):
        pred = pred.sigmoid() > threshold
        return pred


    def get_bev_data(scene_id: str, frame_token: str, cvt_labels_path='/root/autodl-tmp/cvt_labels_nuscenes_v2/'):
        # scene_id = xxxx
        # frame_idx = [0, 40]
        json_file = open(cvt_labels_path + scene_id + '.json')
        scene_data = json.load(json_file)
        for data in scene_data:
            if data['token'] == frame_token:
                break
        bev = Image.open(cvt_labels_path + scene_id + '/' + data['bev'])
        shift = np.arange(12, dtype=np.int32)[None, None]
        bev = np.array(bev)[..., None]
        bev = (bev >> shift) & 1
        # extract road bev
        bev = bev[:, :, 1] | bev[:, :, 0]
        return bev


    dataset = MVResultDataset(os.path.join(args.output_dir, args.exp_name, 'test_results'))


    def cf(batch_data):
        # batch_size=1 is required. because these results are stored as a batch.
        return batch_data[0]


    # keep num_workers = 0
    data_loader = torch.utils.data.DataLoader(dataset, num_workers=0, batch_size=1, collate_fn=cf)

    for batch_i, batch in enumerate(tqdm.tqdm(data_loader)):
        # get masked gt
        images_gt = rearrange(torch.tensor(batch['images_gt']).cuda(), 'b m h w c -> (b m) c h w')
        images_gen = rearrange(torch.tensor(batch['images_gen']).cuda(), 'b m h w c -> (b m) c h w')
        # fid
        _half_num = images_gt.shape[0] // 2
        fid_gt.update(images_gt[:_half_num], real=True)
        fid_gt.update(images_gt[_half_num:], real=False)

        if not REMOVE_MASK:
            fid.update(images_gt, real=True)

        if ADD_MASK:
            obj_mask = images_gt != 0
            fid.update(images_gen * obj_mask, real=False)
        else:
            fid.update(images_gen, real=False)
        # IS
        inception.update(images_gen)
        inception_gt.update(images_gt)

        # bev_IoU
        # CVT cam order
        cam_list = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
        raw_datas = [torch.load(path) for path in batch['raw_data_path']]

        if REMOVE_MASK:
            # update FID gt data here
            _images_gt = np.stack([
                np.stack([cv2.resize(data[cam]['rgb_image'], (448, 256)) for cam in cam_list]) for data in raw_datas
            ], axis=0)
            fid.update(torch.tensor(rearrange(_images_gt, 'b m h w c -> (b m) c h w')).cuda(), real=True)

        datas = [{
            'scene_id': data['meta']['scene_name'],
            'token': data['token'],
            'rotation_quat': [data[cam]['cam_params']['rotation'] for cam in cam_list],
            'cam_trans': [data[cam]['cam_params']['translation'] for cam in cam_list],
        } for data in raw_datas]

        # prepare CVT inputs
        cvt_input_imgs = []

        for b_i in range(batch['images_gen'].shape[0]):
            cvt_input_imgs.append(np.stack(
                [cv2.resize(batch['images_gen'][b_i, m_i], (480, 224)) for m_i in [0, 1, 2, 5, 4, 3]]
                # cvt cam order
            ))

        cvt_input_imgs = torch.tensor(np.stack(cvt_input_imgs) / 255.0, dtype=torch.float32, device='cuda')

        cvt_batch = {
            'image': rearrange(cvt_input_imgs, 'b m h w c -> b m c h w'),  # b m c h w, cuda
            'intrinsics': cvt_cam_intrin[None].repeat(cvt_input_imgs.shape[0], 1, 1, 1).cuda(),
            'extrinsics': torch.stack(
                [torch.tensor(get_extrinsic(data['rotation_quat'], data['cam_trans'])) for data in datas]
            ).transpose_(-1, -2).cuda()
        }

        with torch.no_grad():
            pred = cvt_model(cvt_batch)
            # (b, m, 200, 200)
            bev_pred = post_process_cvt_pred(pred['bev'][:, 0]).to(torch.bool)
            # BEV size, Ours: 80m x 80m, pretrained CVT: 100m x 100m, so do crop.
            _crop = CenterCrop((160, 160))
            bev_gt = torch.stack([
                torch.tensor(get_bev_data(data['scene_id'], data['token']), dtype=torch.bool, device='cuda')
                for data in datas
            ])
            bev_IoU.update(_crop(bev_pred), _crop(bev_gt))

        # Prepare masked PSNR
        _, c, h, w = images_gt.shape
        correspondences = get_correspondences(batch['homos'], h, w, batch['homos'].device)
        correspondences[..., 0] = correspondences[..., 0] / (w - 1) * 2 - 1
        correspondences[..., 1] = correspondences[..., 1] / (h - 1) * 2 - 1
        masks = ((correspondences > -1) & (correspondences < 1)).all(-1)
        m = correspondences.shape[1]
        for i in range(m):
            j = (i + 1) % m
            xy_l = correspondences[:, j, i]
            mask = masks[:, j, i]

            image_src = rearrange(images_gen, '(b m) c h w -> b m c h w', m=m)[:, i].float() / 255
            image_warp = F.grid_sample(image_src, xy_l, align_corners=True)
            image_tgt = rearrange(images_gen, '(b m) c h w -> b m c h w', m=m)[:, j].float() / 255
            mask = mask.unsqueeze(1).expand(-1, 3, -1, -1)
            psnr.update(image_warp[mask], image_tgt[mask])

            # psnr gt
            image_src_gt = rearrange(images_gt, '(b m) c h w -> b m c h w', m=m)[:, i].float() / 255
            image_warp_gt = F.grid_sample(image_src_gt, xy_l, align_corners=True)
            image_tgt_gt = rearrange(images_gt, '(b m) c h w -> b m c h w', m=m)[:, j].float() / 255
            psnr_gt.update(image_warp_gt[mask], image_tgt_gt[mask])

        # cs
        images = batch['images_gen']
        images_gt = batch['images_gt']
        for b in range(images.shape[0]):
            for i in range(images.shape[1]):
                cs.update(torch.tensor(images[b, i]).cuda(), batch['prompt'][b][i])
                cs_gt.update(torch.tensor(images_gt[b, i]).cuda(), batch['prompt'][b][i])

    print(f"FID: {fid.compute()}")
    print(f"FID_GT: {fid_gt.compute()}")
    print(f"IS: {inception.compute()}")
    print(f"IS_GT: {inception_gt.compute()}")
    print(f"CS: {cs.compute()}")
    print(f"CS_GT: {cs_gt.compute()}")
    print(f"PSNR: {psnr.compute()}")
    print(f"PSNR_GT: {psnr_gt.compute()}")
    print(f'IoU_BEV: {bev_IoU.compute()}')