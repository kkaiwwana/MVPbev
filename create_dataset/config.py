# config file for creating custom dataset from NuScenes

KEYWORDS = ['database', '']


database = {
    'dataroot': '../nuscenes/',
    'version': 'v1.0-trainval',
    'verbose': True,
}

map_classes = [
    'drivable_area',
    'ped_crossing',
    'walkway',
    'stop_line',
    'carpark_area',
    'road_divider',
    'lane_divider',
    'road_block',
]

pers_semantic_classes = [
    'drivable_area',
    'road_segment',
    'lane',
    'ped_crossing',
    'walkway',
]

map_radius = 40

# enable this, your dataset will contain rgb images instead of image paths
# if enalbed, it's recommend to enbale in-place resize to save diskspace.
incorporate_image = True
do_resize = True
target_size = {'width': 448, 'height': 256}


# optional, default: False
get_isolated_regions = False

# pack (H, W, 8-map-classes) semantics to (H, W, 1 * Byte), recommend.
pack_pers_semantic_to_byte = True

pack_bev_map_to_byte = False

# while generating, semantic data & ego pose will be rotated (angle in degree) for testing
perspective_semantic_rotation_yaw = 20


enable_segmenter = True
segmenter = {
    'processor_path': '/root/autodl-tmp/pretrained/mask2former',
    'model_path': '/root/autodl-tmp/pretrained/mask2former',
    'device': 'cuda',
    'output_semantic_size': target_size if do_resize else None
}

enable_captioning = True
captioning_model = {
    'processor_path': '/root/autodl-tmp/pretrained/BLIP',
    'model_path': '/root/autodl-tmp/pretrained/BLIP',
    'device': 'cuda',
}



