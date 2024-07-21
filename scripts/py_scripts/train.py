import cv2
import sys
import torch
import argparse
import numpy as np

from tqdm import tqdm
from transformers import logging


parser = argparse.ArgumentParser(prog='Training.')

parser.add_argument('--step', '--s', type=int,
                    help='exp step i, (i = 1, 2) which refers to [1.ft_sd, 3.train_MVAttn]')
parser.add_argument('--exp_name', '--n', type=str, default='default_exp_name')

if __name__ == '__main__':
    sys.path.append('../../')
    from src.dataset import GeneratedNuscDS
    from src.models import MultiViewLDM

    logging.set_verbosity_error()

    arguments = parser.parse_args()

    if arguments.step == 0:
        from configs import finetune_SD
        cfg = finetune_SD
    elif arguments.step == 1:
        from configs import train_MVAttn
        cfg = train_MVAttn
    else:
        raise ValueError

    train_ds = GeneratedNuscDS(cfg.dataset['train'])
    valid_ds = GeneratedNuscDS(cfg.dataset['valid'])

    train_loader = train_ds.get_loader(
        batch_size=cfg.train['batch_size'],
        num_workers=cfg.train['num_workers'],
        drop_last=True,
        shuffle=True,
    )

    valid_loader = valid_ds.get_loader(
        batch_size=cfg.train['batch_size'],
        num_workers=cfg.train['num_workers'],
        drop_last=False,
        shuffle=False
    )

    model = MultiViewLDM(cfg.model)

    if cfg.train['ckpt_path'] is not None:
        model.load_state_dict(torch.load(cfg.train['ckpt_path'], map_location='cuda')['model'], strict=False)

    if cfg.train['optimizer']['name'] == 'Adam':
        param_groups = []
        for params, lr_scale in model.trainable_parameters:
            param_groups.append({"params": params, "lr": cfg.train['optimizer']['lr'] * lr_scale})
        optimizer = torch.optim.Adam(param_groups)
    elif cfg.train['optimizer']['name'] == 'AdamW':
        param_groups = []
        for params, lr_scale in model.trainable_parameters:
            param_groups.append({"params": params, "lr": cfg.train['optimizer']['lr'] * lr_scale})
        optimizer = torch.optim.AdamW(param_groups)
    elif cfg.train['optimizer']['name'] == 'SGD':
        param_groups = []
        for params, lr_scale in model.trainable_parameters:
            param_groups.append({
                'params': params,
                'lr': cfg.train['optimizer']['lr'] * lr_scale,
                'momentum': cfg.train['optimizer']['momentum'],
                'weight_decay': cfg.train['optimizer']['weight_decay'],
            })
        optimizer = torch.optim.SGD(param_groups)
    else:
        raise NotImplementedError

    if cfg.train['lr_scheduler']['name'] == 'CosineAnnealingLR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train['epoch'], eta_min=1e-7)
    else:
        lr_scheduler = None

    # move to device
    device = cfg.train['device']
    model = model.to(device)


    def to_device(target, device='cuda'):
        # move data to device by recursion
        if isinstance(target, torch.Tensor):
            return target.cuda()

        elif isinstance(target, dict):
            for k in target.keys():
                target[k] = to_device(target[k], device)
        elif isinstance(target, list):
            for ele in target:
                ele = to_device(ele, device)
        return target


    with tqdm(total=cfg.train['sanity_check_train'], leave=False) as pbar:
        pbar.set_description(f'train sanity check:')
        for i, batch in enumerate(train_loader):
            batch = to_device(batch, device)
            loss = model.training_step(batch)
            assert loss != torch.nan and loss != torch.inf, f'train sanity check failed, found loss {loss}'
            pbar.set_postfix_str(f'loss: {loss.item():.4f}')
            loss.backward()
            pbar.update(1)
            if i + 1 == cfg.train['sanity_check_train']:
                break

    with tqdm(total=cfg.train['sanity_check_val'], leave=True) as pbar:
        pbar.set_description(f'val sanity check:')
        images = []
        for i, batch in enumerate(valid_loader):
            batch = to_device(batch, device)
            images.append(model.validation_step(batch)[0])

            pbar.update(1)
            if i + 1 == cfg.train['sanity_check_val']:
                break

    # pass sanity check, build exp log
    import os

    exp_name = arguments.exp_name
    log_path = '../../logs/' + exp_name
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/sanity_check_output', exist_ok=True)
    os.makedirs(log_path + '/val_outputs', exist_ok=True)
    os.makedirs(log_path + '/checkpoints', exist_ok=True)
    loss_log = open(log_path + '/loss.txt', mode='w')

    for i, image in enumerate(images):
        for b_i in range(image.shape[0]):
            for m_i in range(image.shape[1]):
                cv2.imwrite(
                    log_path + f'/sanity_check_output/val{i}_b{b_i}_v{m_i}.jpg', image[b_i, m_i][:, :, [2, 1, 0]]
                )

    try:
        loss_monitor = 1e5
        step_cnt = 0
        for epoch in range(cfg.train['epoch']):
            # train steps
            with tqdm(total=len(train_loader), leave=False) as pbar:
                pbar.set_description(f'epoch: {epoch}')
                mean_loss = 0
                for i, batch in enumerate(train_loader):
                    batch = to_device(batch, device)
                    loss = model.training_step(batch)
                    step_cnt += 1
                    mean_loss += loss.item()
                    pbar.set_postfix_str(f'loss: {loss.item():.4f}')
                    if (step_cnt + 1) % cfg.train['log_every_n_steps'] == 0:
                        mean_loss = mean_loss / cfg.train['log_every_n_steps']
                        loss_log.write(f'epoch: {epoch}, step: {step_cnt}, loss: {mean_loss:.4f}\n')
                        if mean_loss < loss_monitor:
                            # save model checkpoint
                            torch.save({
                                'epoch': epoch,
                                'model': model.state_dict(),
                                'optimizer': optimizer.state_dict(),
                                'loss': mean_loss,
                            }, log_path + '/checkpoints/min_loss.ckpt')
                            loss_monitor = mean_loss
                        mean_loss = 0

                    if cfg.train['grad_accu_steps'] > 1:
                        loss /= cfg.train['grad_accu_steps']

                    loss.backward()

                    if cfg.train['grad_accu_steps'] > 1 and step_cnt % cfg.train['grad_accu_steps'] == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                    else:
                        optimizer.step()
                        optimizer.zero_grad()

                    pbar.update(1)

            if lr_scheduler is not None:
                lr_scheduler.step()

            # valid steps
            with tqdm(total=cfg.train['limit_val_batches'], leave=False) as pbar:
                pbar.set_description(f'validating')
                # for i, batch in enumerate(valid_loader):
                for i, batch in enumerate(train_loader):
                    batch = to_device(batch, device)
                    pred, images = model.validation_step(batch)
                    for b_i in range(batch['images'].shape[0]):
                        img_preds = []
                        img_gts = []

                        for m_i in range(batch['images'].shape[1]):
                            image_pred = (
                                    pred[b_i, m_i] * 0.8 + (
                                        batch['semantics'][b_i, m_i].cpu().numpy() + 1) * 127.5 * 0.2
                            ).astype(np.uint8)
                            image_gt = (
                                    images[b_i, m_i] * 0.6 + (
                                        batch['semantics'][b_i, m_i].cpu().numpy() + 1) * 127.5 * 0.4
                            ).astype(np.uint8)

                            img_preds.append(image_pred[:, :, [2, 1, 0]])
                            img_gts.append(image_gt[:, :, [2, 1, 0]])

                        cv2.imwrite(
                            log_path + f'/val_outputs/{step_cnt + 1}_b{i}.jpg',
                            np.concatenate([np.concatenate(img_preds, axis=1), np.concatenate(img_gts, axis=1)], axis=0)
                        )

                    pbar.update(1)
                    if i + 1 == cfg.train['limit_val_batches']:
                        break

    except Exception as e:
        print('error occurred：', e)

    finally:
        loss_log.close()
        torch.save({
            'epoch': -1,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'loss': mean_loss,
        }, log_path + '/checkpoints/last.ckpt')