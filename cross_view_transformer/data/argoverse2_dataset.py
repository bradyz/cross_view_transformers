import json
import torch
import pdb
import pyarrow.feather as feather
from typing import List, Tuple, Any, Dict
from PIL import Image
import numpy as np
import pandas as pd
import pickle

from pathlib import Path
from .common import get_split, INTERPOLATION, get_view_matrix, get_pose
from .transforms import Sample, LoadDataTransform, SaveDataTransform

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.constants import RingCameras, AnnotationCategories
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap, DrivableAreaMapLayer
from av2.geometry.se3 import SE3
from av2.geometry.geometry import compute_interior_points_mask
from av2.rendering.rasterize import xyz_to_bev
from av2.structures.cuboid import CuboidList
from av2.utils.io import read_city_SE3_ego, read_img
import random


CLASS_TO_LAYER_INDEX: Dict[Any,int] = {
    "roi": 0,
    "drivable_area": 1,
    "lane_boundary": 2,
    AnnotationCategories.RAILED_VEHICLE : 3,
    AnnotationCategories.REGULAR_VEHICLE : 3,
    AnnotationCategories.LARGE_VEHICLE : 3,
    
    AnnotationCategories.BOX_TRUCK : 4,
    AnnotationCategories.TRUCK_CAB : 4,
    AnnotationCategories.TRUCK : 4,

    AnnotationCategories.ARTICULATED_BUS : 5,
    AnnotationCategories.BUS : 5,
    AnnotationCategories.SCHOOL_BUS : 5,

    AnnotationCategories.VEHICULAR_TRAILER : 6,
    
    AnnotationCategories.BOLLARD: 7,
    AnnotationCategories.CONSTRUCTION_BARREL : 7,
    AnnotationCategories.CONSTRUCTION_CONE : 7,
    
    AnnotationCategories.PEDESTRIAN : 8,
    AnnotationCategories.OFFICIAL_SIGNALER : 8,
    AnnotationCategories.STROLLER : 8,
    AnnotationCategories.WHEELCHAIR : 8,

    AnnotationCategories.MOTORCYCLIST : 9,
    AnnotationCategories.MOTORCYCLE : 9,
    
    AnnotationCategories.WHEELED_RIDER : 10,
    AnnotationCategories.WHEELED_DEVICE : 10,
    AnnotationCategories.BICYCLE : 10,
    AnnotationCategories.BICYCLIST : 10,    

    # NOT INCLUDED AT THE MOMENT
    # AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN : 1,
    # AnnotationCategories.SIGN : 1,
    # AnnotationCategories.STOP_SIGN : 1,
    # AnnotationCategories.TRAFFIC_LIGHT_TRAILER : 1,
    # AnnotationCategories.MESSAGE_BOARD_TRAILER : 1,
    # AnnotationCategories.ANIMAL : 1,
    # AnnotationCategories.DOG : 1,
}
NUM_LAYERS = max(CLASS_TO_LAYER_INDEX.values())+1
COLORING = {
    0: (0,0,0),
    1: (105,105,105),
    2: (0,0,0),
    3: (35,13,232),
    4: (157,30,150),
    5: (251,207,34),
    6: (138,89,196),
    7: (255,147,29),
    8: (189,142,0),
    9: (113,182,255),
    10: (133,41,107),
}

def get_data(
    dataset_dir,
    labels_dir,
    split,
    version,
    num_classes,
    augment="none",
    image=None,  # image config
    dataset="unused",  # ignore
    **dataset_kwargs,
):
    dataset_dir = Path(dataset_dir)
    labels_dir = Path(labels_dir)

    log_ids = get_split(f"mini_{split}" if version == "mini" else split, "argoverse2") # TODO: modify for argopaths (actually this seems fine)

    sensor_dataloader = SensorDataloader(
            dataset_dir=dataset_dir,
            with_annotations=True,
            with_cache=False
        )
    
    for log_id in log_ids:
        assert Path.exists(Path(dataset_dir, split, log_id))

    return [Argoverse2Dataset(
        log_id=log_id,
        split=split,
        log_specific_dataset_path=Path(dataset_dir,split,log_id),
        synchronized_timestamps=
            sensor_dataloader.synchronization_cache.loc[(split, log_id, 'lidar')][['lidar']+[cam.value for cam in tuple(RingCameras)]],
        transform=SaveDataTransform(labels_dir), # TODO: Transform / SaveDataTransform
        ) for log_id in log_ids]


class Argoverse2Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                    log_id: str, 
                    split: str,
                    log_specific_dataset_path: Path,
                    synchronized_timestamps: pd.DataFrame,
                    transform=None,
                    bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
                ):

        print(f'Argoverse2Dataset building: {log_id}')
        self.log_id = log_id
        self.log_specific_dataset_path = log_specific_dataset_path
        self.poses = read_city_SE3_ego(self.log_specific_dataset_path)
        self.cameras = [PinholeCamera.from_feather(self.log_specific_dataset_path, cam.value) for cam in tuple(RingCameras)]
        self.avm = ArgoverseStaticMap.from_map_dir(Path(self.log_specific_dataset_path, 'map'), build_raster=True)
        self.synchronized_timestamps = synchronized_timestamps
        self.transform = transform

        self.view = get_view_matrix(**bev)
        self.bev_info = bev
        self.bev_shape = (bev['h'], bev['w'])
        self.intrinsics = {cam.cam_name:cam.intrinsics for cam in self.cameras}
        self.extrinsics = {cam.cam_name:cam.extrinsics for cam in self.cameras}

        # for all synchronized scenes get the following data
        self.samples = self.get_samples()
        # with open('/srv/share2/apatni30/tmp/AV2_dataset.pkl', 'wb') as pickle_file:
        #     pickle.dump(self, pickle_file)
        #     raise Exception("asdf")

    def get_samples(self) -> List[Dict[str, Any]]:
        data = []
        for i in range(len(self.synchronized_timestamps)):
            data.append(
                {
                'scene': self.log_id,
                'token': self.synchronized_timestamps.iloc[i]['lidar'].value,
                'pose': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value],
                # 'pose_inverse': self.poses[self.synchronized_timestamps.iloc[i]['lidar'].value].inverse(),
                'cam_ids': range(0,len(self.cameras)),
                'cam_channels': [cam.cam_name for cam in self.cameras],
                'intrinsics': self.intrinsics,
                'extrinsics': self.extrinsics,
                'images': [
                            Path(self.log_specific_dataset_path, 'sensors', 'cameras', camera.cam_name, f'{self.synchronized_timestamps.iloc[i][camera.cam_name].value}.jpg') for camera in self.cameras
                        ] # these are image paths
                }
            )
        return data

    def point_cloud_to_bev(
        self,
        points: np.ndarray,
        layers_masks: np.ndarray,
        coloring: Dict[int, Tuple],
        img_size: Tuple[int, int]
        ) -> Tuple[np.ndarray, np.ndarray]:
        """Creates a 2D BEV image from the point cloud.
        
        Note that the order of layers determines which color is on top

        Args:
            points (np.ndarray): (N,3)
            layers_masks (np.ndarray): (L,N)
            coloring (Dict[int, Tuple]): Each layer's respective color (r,g,b)
            img_size (Tuple): (height, width)

        Returns:
            Tuple[np.ndarray, np.ndarray]: image of size (H,W, 3) AND binary masks for each layer (L, H, W)
        """       
        assert points.shape[-1] == 3
        assert points.shape[0] == layers_masks.shape[-1]
        x_points = points[:, 0].astype(np.float32)
        y_points = points[:, 1].astype(np.float32)

        min_x = np.min(x_points)
        min_y = np.min(y_points)
        max_x = np.max(x_points)
        max_y = np.max(y_points)

        # Ensures that the grid is equally wide on each side
        # This is not particularly necessary, but it's a good 
        # check for this specific use case
        assert abs(min_x) == abs(max_x) 
        assert abs(min_y) == abs(max_y)
        

        # Left/Upper Adjust all the points to be in the image coordinate from
        x_points -= min_x
        y_points -= min_y
        
        # Scale the point cloud onto the image plane 
        res_x = (max_x - min_x)/(img_size[1]-1)
        res_y = (max_y - min_y)/(img_size[0]-1)

        x_img = (-y_points / res_y).astype(np.int32)  # x axis is -y in LIDAR
        y_img = (-x_points / res_x).astype(np.int32)  # y axis is -x in LIDAR

        # Shift Pixels to upper left corner (after division)
        # floor & ceil used to prevent anything being rounded to below 0 after shift
        x_img -= int(np.floor(np.min(x_img)))
        y_img -= int(np.ceil(np.min(y_img)))
        
        # Build Image and Binary Semantic Segmentation Masks
        im = np.zeros([img_size[0], img_size[1], 3], dtype=np.uint8)
        binary_semantic_seg = np.zeros((len(coloring), img_size[0], img_size[1]))

        # Fill in pixels for image/mask
        for i in range(len(coloring)):
            im[y_img[layers_masks[i]==1], x_img[layers_masks[i]==1]] = coloring[i]
            binary_semantic_seg[i, y_img[layers_masks[i]==1], x_img[layers_masks[i]==1]] = 1

        return im, binary_semantic_seg


    def generate_bev(self, pose: SE3, timestamp: int) -> Tuple[np.ndarray, np.ndarray]:
        xpoints = np.linspace(-self.bev_info['h_meters']/2,self.bev_info['h_meters']/2,num=self.bev_info['h'])
        ypoints = np.linspace(-self.bev_info['w_meters']/2,self.bev_info['w_meters']/2,num=self.bev_info['w'])
        zpoints = np.linspace(-1,4, num=15)
        
        # note: may have to change map() to list() if numpy upgraded >= 1.16
        points_xy_wrt_src = np.vstack(map(np.ravel, np.meshgrid(xpoints, ypoints, zpoints))).T
        points_xy_wrt_city = pose.transform_point_cloud(points_xy_wrt_src)
        # TODO: consider removing all non-ROI points (assuming everything else works this can save on computation) 
        
        output = np.zeros((NUM_LAYERS, self.bev_info['h'], self.bev_info['w']))
        layer_masks = np.zeros((NUM_LAYERS, points_xy_wrt_city.shape[0]))
        
        drivable_area_raster = self.avm.raster_drivable_area_layer.get_raster_values_at_coords(points_xy_wrt_city, 0)
        layer_masks[1,drivable_area_raster==1] = 1

        for annotation in CuboidList.from_feather(Path(self.log_specific_dataset_path, 'annotations.feather')):
            if annotation.timestamp_ns == timestamp and annotation.category in CLASS_TO_LAYER_INDEX:
                _, is_interior = annotation.compute_interior_points(points_xy_wrt_src)          
                layer_masks[CLASS_TO_LAYER_INDEX[annotation.category], is_interior] = 1

        image_bev, binary_semantic_masks = self.point_cloud_to_bev(points_xy_wrt_src,layer_masks,COLORING,(self.bev_info['h'],self.bev_info['w']))

        Image.save(image_bev, f"/srv/share2/apatni30/cvt_labels_argoverse2/{self.log_id}_{timestamp}.png")

        return image_bev, binary_semantic_masks

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Additional labels for vehicles only.
        # aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)
        bev_img, bev_masks = self.generate_bev(sample['pose'], sample['token'])
        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev_masks,
            # aux=aux,
            # visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
