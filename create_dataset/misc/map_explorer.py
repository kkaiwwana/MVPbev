import math
import descartes

import numpy as np
import matplotlib.pyplot as plt

from typing import *
from PIL import Image
from pyquaternion import Quaternion
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from nuscenes.map_expansion.map_api import NuScenesMapExplorer
from shapely.geometry import Polygon, MultiPolygon, LineString, Point, box


class MapExplorer(NuScenesMapExplorer):
    """
    override render_map_in_image(*) method to get smeantic map in pers-view.
    """
    @staticmethod
    def _get_img_from_fig(fig):
        fig.canvas.draw()
        rgba_buf = fig.canvas.buffer_rgba()
        (w, h) = fig.canvas.get_width_height()
        rgba_arr = np.frombuffer(rgba_buf, dtype=np.uint8).reshape((h, w, 4))[:, :, :3]
        return rgba_arr
    
    def render_map_in_image(self,
                        nusc: NuScenes,
                        sample_token: str,
                        camera_channel: str = 'CAM_FRONT',
                        alpha: float = 0.3,
                        pack_results_to_byte = True,
                        yaw_rotation_angle=0,
                        patch_radius: float = 80,
                        min_polygon_area: float = 2000,
                        render_behind_cam: bool = True,
                        render_outside_im: bool = True,
                        layer_names: List[str] = None,
                        verbose: bool = False,
                        out_path: str = None):
        """
        Render a nuScenes camera image and overlay the polygons for the specified map layers.
        Note that the projections are not always accurate as the localization is in 2d.
        :param nusc: The NuScenes instance to load the image from.
        :param sample_token: The image's corresponding sample_token.
        :param camera_channel: Camera channel name, e.g. 'CAM_FRONT'.
        :param alpha: The transparency value of the layers to render in [0, 1].
        :param patch_radius: The radius in meters around the ego car in which to select map records.
        :param min_polygon_area: Minimum area a polygon needs to have to be rendered.
        :param render_behind_cam: Whether to render polygons where any point is behind the camera.
        :param render_outside_im: Whether to render polygons where any point is outside the image.
        :param layer_names: The names of the layers to render, e.g. ['lane'].
            If set to None, the recommended setting will be used.
        :param verbose: Whether to print to stdout.
        :param out_path: Optional path to save the rendered figure to disk.
        """
        
        near_plane = 1e-8

        if verbose:
            print('Warning: Note that the projections are not always accurate as the localization is in 2d.')

        # Default layers.
        if layer_names is None:
            layer_names =  ['road_segment', 'lane', 'ped_crossing', 'walkway', 'stop_line', 'carpark_area']

        # Check layers whether we can render them.
        for layer_name in layer_names:
            assert layer_name in self.map_api.non_geometric_polygon_layers, \
                'Error: Can only render non-geometry polygons: %s' % layer_names

        # Check that NuScenesMap was loaded for the correct location.
        sample_record = nusc.get('sample', sample_token)
        scene_record = nusc.get('scene', sample_record['scene_token'])
        log_record = nusc.get('log', scene_record['log_token'])
        log_location = log_record['location']
        assert self.map_api.map_name == log_location, \
            'Error: NuScenesMap loaded for location %s, should be %s!' % (self.map_api.map_name, log_location)

        # Grab the front camera image and intrinsics.
        cam_token = sample_record['data'][camera_channel]
        cam_record = nusc.get('sample_data', cam_token)
        cam_path = nusc.get_sample_data_path(cam_token)
        im = Image.open(cam_path)
        im_size = im.size
        cs_record = nusc.get('calibrated_sensor', cam_record['calibrated_sensor_token'])
        
        
        cam_intrinsic = np.array(cs_record['camera_intrinsic'])

        # Retrieve the current map.
        poserecord = nusc.get('ego_pose', cam_record['ego_pose_token'])
        ego_pose = poserecord['translation']
        box_coords = (
            ego_pose[0] - patch_radius,
            ego_pose[1] - patch_radius,
            ego_pose[0] + patch_radius,
            ego_pose[1] + patch_radius,
        )
        records_in_patch = self.get_records_in_patch(box_coords, layer_names, 'intersect')
        
        semantic_data = {}
    
        # Retrieve and render each record.
        for layer_name in layer_names:
            # Init axes.
            fig = plt.figure(figsize=(16, 9), dpi=100)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(0, im_size[0])
            ax.set_ylim(0, im_size[1])
            # ax.imshow(im)
            # ax.imshow(np.zeros((im_size[1], im_size[0])))
            
            for token in records_in_patch[layer_name]:
                record = self.map_api.get(layer_name, token)
                if layer_name == 'drivable_area':
                    polygon_tokens = record['polygon_tokens']
                else:
                    polygon_tokens = [record['polygon_token']]

                for polygon_token in polygon_tokens:
                    polygon = self.map_api.extract_polygon(polygon_token)

                    # Convert polygon nodes to pointcloud with 0 height.
                    points = np.array(polygon.exterior.xy)
                    points = np.vstack((points, np.zeros((1, points.shape[1]))))

                    # Transform into the ego vehicle frame for the timestamp of the image.
                    points = points - np.array(poserecord['translation']).reshape((-1, 1))
                    points = np.dot(Quaternion(poserecord['rotation']).rotation_matrix.T, points)

                    # Transform into the camera.
                    points = points - np.array(cs_record['translation']).reshape((-1, 1))
                    # return Quaternion(cs_record['rotation']).rotation_matrix
                    
                    # rotate yaw for testing
                    beta = yaw_rotation_angle
                    sin_beta = math.sin(beta / 180 * math.pi)
                    cos_beta = math.cos(beta / 180 * math.pi)
                    rot_mat = np.array([[cos_beta, 0, sin_beta], [0, 1, 0], [-sin_beta, 0, cos_beta]])
                    
                    points = np.dot(rot_mat @ Quaternion(cs_record['rotation']).rotation_matrix.T, points)

                    # Remove points that are partially behind the camera.
                    depths = points[2, :]
                    behind = depths < near_plane
                    if np.all(behind):
                        continue

                    if render_behind_cam:
                        # Perform clipping on polygons that are partially behind the camera.
                        points = NuScenesMapExplorer._clip_points_behind_camera(points, near_plane)
                    elif np.any(behind):
                        # Otherwise ignore any polygon that is partially behind the camera.
                        continue

                    # Ignore polygons with less than 3 points after clipping.
                    if len(points) == 0 or points.shape[1] < 3:
                        continue

                    # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                    points = view_points(points, cam_intrinsic, normalize=True)

                    # Skip polygons where all points are outside the image.
                    # Leave a margin of 1 pixel for aesthetic reasons.
                    inside = np.ones(points.shape[1], dtype=bool)
                    inside = np.logical_and(inside, points[0, :] > 1)
                    inside = np.logical_and(inside, points[0, :] < im.size[0] - 1)
                    inside = np.logical_and(inside, points[1, :] > 1)
                    inside = np.logical_and(inside, points[1, :] < im.size[1] - 1)
                    if render_outside_im:
                        if np.all(np.logical_not(inside)):
                            continue
                    else:
                        if np.any(np.logical_not(inside)):
                            continue

                    points = points[:2, :]
                    points = [(p0, p1) for (p0, p1) in zip(points[0], points[1])]
                    polygon_proj = Polygon(points)

                    # Filter small polygons
                    if polygon_proj.area < min_polygon_area:
                        continue

                    label = layer_name
                    _polygon = descartes.PolygonPatch(polygon_proj, fc=self.color_map[layer_name], alpha=alpha,
                                                        label=label)
                    ax.add_patch(_polygon)

            # Display the image.
            plt.axis('off')
            ax.invert_yaxis()

            if out_path is not None:
                plt.tight_layout()
                plt.savefig(out_path, bbox_inches='tight', pad_inches=0)
   
            array_fig = self._get_img_from_fig(fig)

            plt.delaxes()
            plt.close()

            # layer_semantic = array_fig
            if (im_size[1], im_size[0]) != (900, 1600):
                array_fig = cv2.resize(array_fig, (900, 1600), interpolation=cv.Nearest)         
            semantic_data[layer_name] = array_fig
        
        semantic_data['perspective'] = np.stack(
            # IDK why I did "this"(< 255 * 3), but I'd better not modify it, 24.3.28.
            # Okay bro, I get it. you wanna filter out those (255, 255, 255): white background points, 24.3.29
            [(layer_semantic.sum(axis=-1) < 765).astype(np.uint8) for layer_semantic in semantic_data.values()], axis=-1)
        
        if pack_results_to_byte:
            # pack results (H, W, 8-classes) of dtype: uint8 -> (H, W, 1)
            semantic_data['perspective'] = np.packbits(semantic_data['perspective'], axis=-1)
        
        # remove those redudant single layer data
        for layer_name in layer_names:
            semantic_data[layer_name] = None
            semantic_data.pop(layer_name)
            
        return semantic_data