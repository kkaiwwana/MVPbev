import cv2
import torch
import numpy as np

from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_segmentation_masks
from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation


CITYSPACE_LABELS = [
    "road", "sidewalk", "building", "wall", "fence",
    "pole", "traffic light", "traffic sign", "vegetation",
    "terrain", "sky", "person", "rider", "car", "truck",
    "bus", "train", "motorcycle", "bicycle"
]

class NuSceneSegment:
    # categories for pretrained segment model in CitySpace
    id2label = {i: label for i, label in enumerate(CITYSPACE_LABELS)}

    def __init__(self, processor_path, model_path, device='cuda', **kwargs):
        self.processor = AutoImageProcessor.from_pretrained(processor_path)
        self.model = Mask2FormerForUniversalSegmentation.from_pretrained(model_path)
        self.model = self.model.to(device)
        self.device = device
        
        if 'output_semantic_size' in kwargs.keys() and kwargs['output_semantic_size'] != None:
            self.output_size = kwargs['output_semantic_size']    
        else:
            self.output_size = None
    
    @staticmethod
    def get_objects(semantic_map):
        # get objects in the scene, e.g. ['road', 'sidewalk', 'building', ...]
        objects = np.unique(semantic_map)
        return [NuSceneSegment.id2label[object_id.item()] for object_id in objects]
    
    
    @staticmethod
    def convert_CHW_mask(semantic_map):
        return torch.stack([semantic_map == cat_id for cat_id in NuSceneSegment.id2label.keys()])
    
    
    @staticmethod
    def draw_seg_mask(img, mask, alpha=0.6, show=True):
        if mask.dtype is not torch.bool:
            mask = NuSceneSegment.convert_CHW_mask(mask)
        
        fig = draw_segmentation_masks(transforms.PILToTensor()(img), mask, alpha)
        if show:
            fig = fig.permute([1, 2, 0]).numpy()
            plt.figure(figsize=(16, 9))
            plt.imshow(fig)
        
        return fig
    
    
    @staticmethod
    def get_isolated_regions(semantic_map, min_region_pix=144):
        #  return coordinates of points in isolated regions (e.g. region of background buildings)
        
        try:
            from CythonBFS import non_recur_bfs
        except ImportError:
            print('Please build cython-bfs dependency at first. Check `./CythonBFS` for more info.')
        
        
        MAP_RESIZE = 200
        w, h = int(semantic_map.shape[1] / semantic_map.shape[0] * MAP_RESIZE), MAP_RESIZE
        semantic_map = cv2.resize(semantic_map, dsize=(w, h)).astype(np.int64)
        regions = {'resized_semantic': semantic_map}
        
        visited = np.zeros_like(semantic_map)
        
        while (visited == 0).any():
            for i, j in zip(*np.nonzero(visited - 1)):
                if visited[i, j] != 1:
                    region_points = non_recur_bfs(
                        semantic_map, 
                        x=i, y=j, 
                        key=semantic_map[i, j], 
                        visited=visited, 
                        max_len=semantic_map.shape[0] * semantic_map.shape[1]
                    )
                    region_label = NuSceneSegment.id2label[semantic_map[i, j].item()]
                    
                    # ignore small regions
                    if len(region_points) < min_region_pix:
                        continue
                        
                    region_points = np.transpose(region_points)

                    if region_label not in regions.keys():
                        regions[region_label] = [region_points]
                    else:
                        regions[region_label].append(region_points)

                    break
                    
        return regions

    def __call__(self, img: Image or np.ndarray):
        inputs = self.processor(img, return_tensors="pt")
        for key in inputs.keys():
            inputs[key] = inputs[key].to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            
        class_queries_logits = outputs.class_queries_logits
        predicted_semantic_map = self.processor.post_process_semantic_segmentation(
            outputs, target_sizes=[img.size[::-1]])[0]
        
        out = predicted_semantic_map.cpu().numpy().astype(np.uint8)
        
        if self.output_size is not None:
            w, h = self.output_size['width'], self.output_size['height']
            out = cv2.resize(out, dsize=(w, h), interpolation = cv2.INTER_NEAREST)
        
        return out