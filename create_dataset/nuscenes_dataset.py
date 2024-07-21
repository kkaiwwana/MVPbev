import cv2
import numpy as np

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import BoxVisibility
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion
from PIL import Image

from misc import NuSceneCaptioning
from misc import NuSceneSegment
from misc import MapExplorer


class NuScenesDS:
    def __init__(self, config, nusc: NuScenes = None):
        
        self.config = config
        self.dataroot = config.database['dataroot']
        if nusc is None:
            self.nusc = NuScenes(
                dataroot=self.dataroot, 
                version=config.database['version'], 
                verbose=config.database['verbose']
            )
        else:
            # load nusc index in advance to save time
            self.nusc = nusc
        
        self.cameras = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_BACK_RIGHT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_FRONT_LEFT'
        ]
        self.maps = {
            'singapore-onenorth': NuScenesMap(dataroot=self.dataroot, map_name='singapore-onenorth'),
            'singapore-hollandvillage': NuScenesMap(dataroot=self.dataroot, map_name='singapore-hollandvillage'),
            'singapore-queenstown': NuScenesMap(dataroot=self.dataroot, map_name='singapore-queenstown'),
            'boston-seaport': NuScenesMap(dataroot=self.dataroot, map_name='boston-seaport'),
        }
        self.maps_explorer = {
            'singapore-onenorth': MapExplorer(self.maps['singapore-onenorth']),
            'singapore-hollandvillage': MapExplorer(self.maps['singapore-hollandvillage']),
            'singapore-queenstown': MapExplorer(self.maps['singapore-queenstown']),
            'boston-seaport': MapExplorer(self.maps['boston-seaport']),
        }
        
        self.map_layer_classes = config.map_classes
        self.pers_layer_classes = config.pers_semantic_classes
        self.seman_radius = config.map_radius
        
        self.segment_model = NuSceneSegment(**config.segmenter) if config.enable_segmenter else None
        self.captioning_model = NuSceneCaptioning(**config.captioning_model) if config.enable_captioning else None
        
        
    @staticmethod
    def _select_frams(frames, min_gap_distance=20):
        # filter frames based on distance estimation (dist = v * delta_t) 
        res = []
        dis_accu = min_gap_distance
        
        for frame in frames:
            dis_accu += frame['velocity'] * 0.5
            if dis_accu >= min_gap_distance:
                dis_accu -= min_gap_distance
                res.append(frame)
        
        return res
    
    
    def _get_scene_frames(self, head_sample):  
        DELTA_T = 0.5
        
        samples = []
        _sample = head_sample
        
        frame_cnt = 0
        while _sample['next'] != '':
            _next_sample = self.nusc.get('sample', token=_sample['next'])
            
            sample_data = self.nusc.get('sample_data', _sample['data']['CAM_FRONT'])
            next_sample_data = self.nusc.get('sample_data', _next_sample['data']['CAM_FRONT'])
            
            ego_pose = self.nusc.get('ego_pose', sample_data['ego_pose_token'])
            next_ego_pose = self.nusc.get('ego_pose', next_sample_data['ego_pose_token'])
            
            loc = ego_pose['translation']
            next_loc = next_ego_pose['translation']
            
            velocity = sum([(coor_next - coor) ** 2 for coor_next, coor in zip(next_loc, loc)]) ** 0.5 / DELTA_T
            
            samples.append({
                'token': _sample['token'],
                'frame': frame_cnt,
                'velocity': velocity,
            })
            
            frame_cnt += 1
            _sample = _next_sample
        
        return samples
    
    def _get_map_name(self, frame_token):
        return self.nusc.get(
            'log', self.nusc.get(
                'scene', self.nusc.get(
                    'sample', frame_token)['scene_token'])['log_token'])['location']
    
    
    @staticmethod
    def _get_box_2d(self, view: np.ndarray = np.eye(3), normalize: bool = False):

        corners = view_points(self.corners(), view, normalize=normalize)[:2, :]
        points = []

        def get_rect(selected_corners):
            _points = []
            prev = selected_corners[-1]
            for corner in selected_corners:

                _points.append((int(prev[0]), int(prev[1])))
                _points.append((int(corner[0]), int(corner[1])))
                prev = corner

            return _points

        # Draw the sides
        for i in range(4):
            points.append((int(corners.T[i][0]), int(corners.T[i][1])))
            points.append((int(corners.T[i + 4][0]), int(corners.T[i + 4][1])))


        # Draw front (first 4 corners) and rear (last 4 corners) rectangles(3d)/lines(2d)
        points += get_rect(corners.T[:4])
        points += get_rect(corners.T[4:])

        # Draw line indicating the front
        center_bottom_forward = np.mean(corners.T[2:4], axis=0)
        center_bottom = np.mean(corners.T[[2, 3, 7, 6]], axis=0)
        points.append((int(center_bottom[0]), int(center_bottom[1])))
        points.append((int(center_bottom_forward[0]), int(center_bottom_forward[1])))

        points = np.array(points)

        return np.concatenate([points.min(0), points.max(0)], axis=0)
    
    def _get_box_3d(self, frame_token, box_type='LIDAR_TOP'):
        # get 3d gt bboxes w.r.t LIDAR_TOP
        sample_data_token = self.nusc.get('sample', frame_token)['data'][box_type]
        
        sd_rec = self.nusc.get('sample_data', sample_data_token)

        s_rec = self.nusc.get('sample', sd_rec['sample_token'])

        cs_rec = self.nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_rec = self.nusc.get('ego_pose', sd_rec['ego_pose_token'])

        ann_recs = [self.nusc.get('sample_annotation', token) for token in s_rec['anns']]
        
        # mmdet3d setting
        N_CLASSES = 10
        CLASSES = [
            'car',
            'truck',
            'construction',
            'bus',
            'trailer',
            'barrier',
            'motorcycle',
            'bicycle',
            'pedestrian',
            'trafficcone',
        ]
        
        boxes, labels = [], []
        for ann_rec in ann_recs:
            
            ann_rec['sample_annotation_token'] = ann_rec['token']
            ann_rec['sample_data_token'] = sample_data_token

            # 世界坐标系下的box标注信息
            box = self.nusc.get_box(ann_rec['token'])

            # 从世界坐标系->车身坐标系
            box.translate(-np.array(pose_rec['translation']))
            box.rotate(Quaternion(pose_rec['rotation']).inverse)

            # 从车身坐标系->相机坐标系
            box.translate(-np.array(cs_rec['translation']))
            box.rotate(Quaternion(cs_rec['rotation']).inverse)
            
            if set(CLASSES).intersection(box.name.split('.')) != set():
                labels.append(CLASSES.index(set(CLASSES).intersection(box.name.split('.')).pop()))
            else:
                continue
            
            boxes.append(np.concatenate(
                [box.center, box.wlh, np.array([- box.orientation.yaw_pitch_roll[0] - np.pi / 2])], axis=-1
            ))
            
        gt_anns_3d = {
            'gt_bboxes_3d': np.stack(boxes).astype(np.float32) if len(boxes) > 0 else np.zeros((1, 7)), 
            'gt_labels_3d': np.array(labels) if len(boxes) > 0 else np.array([0,]),
            'ego2sensor_rotation': Quaternion(cs_rec['rotation']).inverse, #
            'ego2sensor_translation': - np.array(cs_rec['translation']),
        }
        return gt_anns_3d
    
    
    def _get_bev_semantic(self, frame_token):
        map_name = self._get_map_name(frame_token)
        cam_data_token = self.nusc.get('sample', frame_token)['data']['CAM_FRONT']
        cam_data = self.nusc.get('sample_data', cam_data_token)
        ego_pose = self.nusc.get('ego_pose', cam_data['ego_pose_token'])
        translation = ego_pose['translation']
        patch_box = (
            translation[0],
            translation[1],
            2 * self.seman_radius,
            2 * self.seman_radius,
        )
        patch_angle = Quaternion(ego_pose['rotation']).yaw_pitch_roll[0] / 3.14159 * 180
        bev_semantic = self.maps_explorer[map_name].get_map_mask(
            patch_box, patch_angle, self.map_layer_classes, (200, 200))
        if self.config.pack_bev_map_to_byte:
            bev_semantic = np.packbits(bev_semantic, axis=0)
        return np.flip(bev_semantic, axis=1)
    
    
    def get_annotations(self, frame_token):
        annotations = {}
        
        # get 2D boxes w.r.t EACH CAM
        for camera in self.cameras:
            cam_data_token = self.nusc.get('sample', frame_token)['data'][camera]
            _, boxes, cam_intrinsic = self.nusc.get_sample_data(cam_data_token, box_vis_level=BoxVisibility.ANY) 
            
            boxes_2d = []
            labels = []
            for box in boxes:
                labels.append(box.name)
                p = NuScenesDS._get_box_2d(box, view=cam_intrinsic, normalize=True)
                boxes_2d.append(p)
        
            boxes_2d = np.stack(boxes_2d) if len(labels) != 0 else None
                 
            annotations[camera] = {'boxes': boxes_2d, 'labels': labels}
        
        # get 3D boxes w.r.t Scene-level(i.e. LIDAR-TOP)
        annotations['3d_anns'] = self._get_box_3d(frame_token)

        return annotations
    
    def _get_objects(self, frame_token):
        if self.segment_model is None:
            return {cam: None for cam in self.cameras}
        
        scene_objects = {}
        for camera in self.cameras:
            cam_data_token = self.nusc.get('sample', frame_token)['data'][camera]
            cam_data = self.nusc.get('sample_data', cam_data_token)
            rgb_image = Image.open(self.nusc.get_sample_data_path(cam_data_token))
            
            semantic_map = self.segment_model(rgb_image)
            
            objects_name = self.segment_model.get_objects(semantic_map)
            if self.config.get_isolated_regions:
                # time-comsuming (even thougn cython impl is 50x faster than python)
                isolated_regions = self.segment_model.get_isolated_regions(semantic_map, 200)
            else:
                isolated_regions = None
            
            scene_objects[camera] = {
                'objects': objects_name,
                'semantic_map': semantic_map,
                'isolated_regions': isolated_regions,
            }
        return scene_objects
            
    
    def get_description(self, frame_token):

        scene_objects = self._get_objects(frame_token)
            
        return scene_objects
        
    
    def get_image(self, frame_token, cam_id, load_img=True, resize=False):
        
        cam_data_token = self.nusc.get('sample', frame_token)['data'][cam_id]
        if load_img:
            rgb_image = Image.open(self.nusc.get_sample_data_path(cam_data_token))
            rgb_image = np.array(rgb_image)
        else:
            return self.nusc.get_sample_data_path(cam_data_token)
        
        if resize:
            rgb_image = cv2.resize(
                rgb_image, dsize=(self.config.target_size['width'], self.config.target_size['height'])
            )
        
        return rgb_image

        
    def get_semantic(self, frame_token):
        # get perspective smenatic
        map_name = self._get_map_name(frame_token)
        nusc_map = self.maps[map_name]
        semantic_data = {'token': frame_token}

        for camera in self.cameras:
            map_explorer = self.maps_explorer[map_name]
            semantics = map_explorer.render_map_in_image(
                self.nusc, 
                frame_token, 
                layer_names=self.pers_layer_classes, 
                patch_radius=self.seman_radius,
                pack_results_to_byte=self.config.pack_pers_semantic_to_byte,
                yaw_rotation_angle=self.config.perspective_semantic_rotation_yaw,
                camera_channel=camera,
            )
            if self.config.do_resize:
                semantics = cv2.resize(
                    semantics['perspective'], 
                    dsize=(self.config.target_size['width'], self.config.target_size['height']),
                    interpolation = cv2.INTER_NEAREST, # ensure packed data safe
                )[:, :, None]
            
            semantic_data[camera] = semantics
            
        return semantic_data
        
    
    def __getitem__(self, idx, min_gap_distance=8):
        scene = self.nusc.scene[idx]
        head_sample = self.nusc.get('sample', token=scene['first_sample_token'])
        scene_frames = self._get_scene_frames(head_sample)
        seleted_frames = NuScenesDS._select_frams(scene_frames, min_gap_distance)
        return seleted_frames
    
    def get_data(self, frames, data_keys=None):
        if data_keys is None:
            data_keys = ['rgb', 'semantic', 'annotations', 'objects', 'prompt', 'bev']
        if not isinstance(frames, list):
            frames = [frames]
        # from ipdb import set_trace as st
        # st()

        for i, frame in enumerate(frames):
            # try:
            frame_token = frame['token']
            timestamp = self.nusc.get('sample', frame_token)['timestamp']

            frame_data = {camera: {} for camera in self.cameras }
            frame_data['token'] = frame_token

            # ego_poses & camera_params
            ego_poses = {}
            camera_params = {}
            for cam_id in self.cameras:
                cam_data_token = self.nusc.get('sample', frame_token)['data'][cam_id]
                cam_data = self.nusc.get('sample_data', cam_data_token)
                frame_data[cam_id]['cam_params'] = self.nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])
                frame_data[cam_id]['ego_pose'] = self.nusc.get('ego_pose', cam_data['ego_pose_token'])

            # rgb iamge
            if 'rgb' in data_keys:
                for cam_id in self.cameras:
                    img = self.get_image(frame_token, cam_id, self.config.incorporate_image, self.config.do_resize)
                    frame_data[cam_id]['rgb_image'] = img
                    if 'prompt' in data_keys:
                        frame_data[cam_id]['prompt'] = self.captioning_model(img)

            # bev_map
            if 'bev' in data_keys:
                frame_data['bev'] = self._get_bev_semantic(frame_token)

            nusc_des = self.nusc.get('scene', self.nusc.get('sample', frame_token)['scene_token'])

            if 'semantic' in data_keys:    
                semantic_data = self.get_semantic(frame_token)
                for camera in self.cameras:
                    frame_data[camera]['semantic'] = semantic_data[camera]

            if 'annotations' in data_keys:
                annotations = self.get_annotations(frame_token)
                for camera in self.cameras:
                    frame_data[camera]['annotations'] = annotations[camera]

                frame_data['3d_anns'] = annotations['3d_anns']

            if 'objects' in data_keys:
                description = self.get_description(frame_token)
                for camera in self.cameras:
                    frame_data[camera]['objects'] = description[camera]

            frame_data['meta'] = {
                'description': nusc_des['description'], 
                'scene_name': nusc_des['name'], 
                'map_name': self._get_map_name(frame_token),
                'timestamp': timestamp,
            }
#             except Exception as err:
#                 print(f'Error occurred: {err}, at frame {frame}.')

#             finally:     
            yield frame_data
        
    def __len__(self):
        return len(self.nusc.scene)