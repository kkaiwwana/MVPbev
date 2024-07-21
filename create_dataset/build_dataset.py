import torch
import config
import pandas as pd
import warnings

from tqdm import tqdm
from nuscenes_dataset import NuScenesDS
from nuscenes.utils.splits import create_splits_scenes


if __name__ == '__main__':
    # todo: improve this script (this version is too simple)
    warnings.filterwarnings('ignore')
    
    ds = NuScenesDS(config)

    train_scene = []
    train_filepath = []
    valid_scene = []
    valid_filepath = []
    
    train_df = pd.DataFrame(columns=['scene', 'filepath'])
    valid_df = pd.DataFrame(columns=['scene', 'filepath'])

    splits = create_splits_scenes()
    
    frames = []
    # try:
    for i in range(len(ds)):
        frames += ds[i]

    with tqdm(total=len(frames)) as pbar:
        for i, data in enumerate(ds.get_data(frames)):
            if data is not None:
                if data['meta']['scene_name'] in splits['train']:  
                    train_scene.append(data['meta']['scene_name'])
                    train_filepath.append('data/' + data['token'] + '.pt')
                else:
                    valid_scene.append(data['meta']['scene_name'])
                    valid_filepath.append('data/' + data['token'] + '.pt')

                torch.save(data, '../GeneratedNusc/data/' + data['token'] + '.pt')
            else:
                continue
            pbar.update(1)
    # except Exception as e:
    #     print(f'Error occurred: {e} at i={i}')
    
    # finally:
    train_df['scene'] = train_scene
    train_df['filepath'] = train_filepath
    train_df.to_csv('../GeneratedNusc/train.csv', header=True, index=False)

    valid_df['scene'] = valid_scene
    valid_df['filepath'] = valid_filepath
    valid_df.to_csv('../GeneratedNusc/valid.csv', header=True, index=False)
    
    
    
    
   
    
    