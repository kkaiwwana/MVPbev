import sys
import torch
import argparse
from tqdm import tqdm


parser = argparse.ArgumentParser(prog='..')
parser.add_argument('--exp_name', '--n', type=str, help='exp name(to be used as dir name)')
parser.add_argument('--output_dir', '--o', type=str, default='../../test_output/')


if __name__ == '__main__':
    sys.path.append('../../')
    from src.dataset import GeneratedNuscDS
    from src.models import MultiViewLDM

    from transformers import logging
    logging.set_verbosity_error()
    
    arguments = parser.parse_args()
    
    try:
        from configs import test_config as cfg
    except ImportError:
        print('ensure your config file in the right dir, which is suppose to be MVPbev/configs/test_config.py')

    valid_ds = GeneratedNuscDS(cfg.dataset['valid'])
    test_loader = valid_ds.get_loader(
        batch_size=cfg.test['batch_size'],
        num_workers=cfg.test['num_workers'],
        drop_last=False,
        shuffle=False
    )

    model = MultiViewLDM(cfg.model)

    assert cfg.test['ckpt_path'] is not None, 'checkpoint is required when test.'
    model.load_state_dict(torch.load(cfg.test['ckpt_path'], map_location='cuda')['model'], strict=False)
    
    # move to device
    device = cfg.test['device']
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
    
    import os
    exp_name = arguments.exp_name
    log_path = arguments.output_dir + exp_name
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(log_path + '/test_results', exist_ok=True)
    
    try:
        with tqdm(total=len(test_loader), leave=False) as pbar:
            pbar.set_description(f'testing')
            for i, batch in enumerate(test_loader):
                batch = to_device(batch, device)
                pred = model.test_step(batch)
                torch.save(pred, log_path + f'/test_results/sample_{i}.pt')
                pbar.update(1)

    except Exception as e:
        print('error occurred：', e)

    finally:
        print('Done.')