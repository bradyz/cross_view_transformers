import json
import torch
import pyarrow.feather as feather
from typing import List, Tuple
import numpy as np

from pathlib import Path
from .common import get_split, INTERPOLATION, get_view_matrix, get_pose
from .transforms import Sample, LoadDataTransform

from av2.datasets.sensor.av2_sensor_dataloader import AV2SensorDataLoader
from av2.datasets.sensor.sensor_dataloader import SensorDataloader
from av2.datasets.sensor.constants import RingCameras, AnnotationCategories
from av2.geometry.camera.pinhole_camera import PinholeCamera
from av2.map.map_api import ArgoverseStaticMap, DrivableAreaMapLayer
from av2.geometry.se3 import SE3
from av2.geometry.geometry import compute_interior_points_mask
from av2.rendering.rasterize import xyz_to_bev

STATIC = [
    AnnotationCategories.BOLLARD ,
    AnnotationCategories.CONSTRUCTION_BARREL ,
    AnnotationCategories.CONSTRUCTION_CONE ,
    AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN ,
    AnnotationCategories.SIGN ,
    AnnotationCategories.STOP_SIGN ,
    AnnotationCategories.TRAFFIC_LIGHT_TRAILER ,
    AnnotationCategories.MOBILE_PEDESTRIAN_CROSSING_SIGN ,

          ]
DYNAMIC = [
    AnnotationCategories.ARTICULATED_BUS ,
    AnnotationCategories.BICYCLE ,
    AnnotationCategories.ANIMAL ,
    AnnotationCategories.BICYCLIST ,
    AnnotationCategories.BOX_TRUCK ,
    AnnotationCategories.BUS ,
    AnnotationCategories.DOG ,
    AnnotationCategories.LARGE_VEHICLE ,
    AnnotationCategories.MESSAGE_BOARD_TRAILER ,
    AnnotationCategories.MOTORCYCLE ,
    AnnotationCategories.MOTORCYCLIST ,
    AnnotationCategories.OFFICIAL_SIGNALER ,
    AnnotationCategories.PEDESTRIAN ,
    AnnotationCategories.BICYCLIST ,
    AnnotationCategories.RAILED_VEHICLE ,
    AnnotationCategories.REGULAR_VEHICLE ,
    AnnotationCategories.SCHOOL_BUS ,
    AnnotationCategories.STROLLER ,
    AnnotationCategories.TRUCK ,
    AnnotationCategories.TRUCK_CAB ,
    AnnotationCategories.VEHICULAR_TRAILER ,
    AnnotationCategories.WHEELCHAIR ,
    AnnotationCategories.WHEELED_RIDER ,
    AnnotationCategories.WHEELED_DEVICE ,
          ]

def get_data(
    dataset_dir,
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

    # Override augment if not training
    augment = "none" if split != "train" else augment
    transform = LoadDataTransform(dataset_dir, image, num_classes, augment)

    # Format the split name
    split = f"mini_{split}" if version == "mini" else split
    scene_ids = get_split(split, "argoverse2")

    av2_dataloader = AV2SensorDataLoader(dataset_dir, dataset_dir)
    
    return [Argoverse2Dataset(
        scene_id = scene_id,
        sensor_dataset_path = Path(dataset_dir, 'sensor', split, scene_id),
        map_path = Path(dataset_dir, 'sensor', split, scene_id, 'map'),
        cameras=[av2_dataloader.get_log_pinhole_camera(scene_id, cam) for cam in tuple(RingCameras)],
        transform=0,
        poses=0,
        ) for scene_id in scene_ids]


class Argoverse2Dataset(torch.utils.data.Dataset):
    
    def __init__(self, 
                    scene_id: str, 
                    sensor_dataset_path: Path,
                    map_path: Path,
                    cameras: List[PinholeCamera],
                    transform=None,
                    bev={'h': 200, 'w': 200, 'h_meters': 100, 'w_meters': 100, 'offset': 0.0},
                ):
        assert sensor_dataset_path.exists()
        self.scene_name = scene_id

        self.sensor_dataloader = SensorDataloader(
            dataset_dir=sensor_dataset_path,
            with_annotations=True,
            with_cache=False
        )
        self.avm = ArgoverseStaticMap.from_map_dir(map_path, build_raster=False)

        self.view = get_view_matrix(**bev)
        self.bev_info = bev
        self.bev_shape = (bev['h'], bev['w'])
        self.intrinsics = {k:v.intrinsics for k,v in cameras.items()}
        self.extrinsics = {k:v.extrinsics for k,v in cameras.items()}

        # for all synchronized scenes get the following data
        self.samples = self.get_samples()

    def get_samples(self):
        data = []
        for sensor_data in self.sensor_dataloader:

            data_i = {
                'scene': self.scene_name,
                'token': sensor_data.timestamp_ns,
                'pose': sensor_data.timestamp_city_SE3_ego_dict[sensor_data.timestamp_ns],
                # 'pose_inverse': egolidarflat_from_world.tolist(),
                'cam_ids': range(0,len(self.cameras)),
                'cam_channels': list(RingCameras),
                'intrinsics': self.intrinsics,
                'extrinsics': self.extrinsics,
                'images': sensor_data.synchronized_imagery,
            }

    def generate_bev(self, pose: SE3) -> Tuple[np.ndarray, np.ndarray]:
        xpoints = np.arange(-self.bev_info['h_meters']/2,self.bev_info['h_meters']/2,self.bev_info['h_meters']/self.bev_info['h'])
        ypoints = np.arange(-self.bev_info['w_meters']/2,self.bev_info['w_meters']/2,self.bev_info['w_meters']/self.bev_info['w'])
        zpoints = np.array([0,.5,1])
        # xcoord, ycoord, z_coord= np.meshgrid(xpoints, ypoints)
        # xyz_coords = np.concatenate((xcoord, ycoord, z_coord),axis=2)
        # points_xy_city = location.transform_point_cloud(xyz_coords)

        # note: may have to change map() to list() if numpy upgraded >= 1.16
        points_xy_wrt_src = np.vstack(map(np.ravel, np.meshgrid(xpoints, ypoints, zpoints))).T
        points_xy_wrt_city = pose.transform_point_cloud(points_xy_wrt_src)
        
        # Project cuboids onto raster
        dynamic_raster = np.zeros(shape=(points_xy_wrt_city.shape[0]))
        static_raster = np.zeros(shape=(points_xy_wrt_city.shape[0]))
        for annotation in self.sensor_dataloader.annotations:
            _, is_interior = annotation.compute_interior_points_mask(points_xy_wrt_city)                
            if annotation.category in STATIC:
                static_raster = static_raster | is_interior
            elif annotation.category in DYNAMIC:
                dynamic_raster = dynamic_raster | is_interior

        # Get driveable area raster
        drivable_area_raster = self.avm.static_map.raster_drivable_area_layer.get_raster_values_at_coords(points_xy_wrt_city, 0)

        drivable_area_raster[np.where(static_raster=1)] = 2
        drivable_area_raster[np.where(dynamic_raster=1)] = 3

        # Convert the Driveable Area Raster + Dynamic + Static Map into a BEV Image
        output_bev = xyz_to_bev(
            drivable_area_raster,
            voxel_resolution=(self.bev_info['h'], self.bev_info['w'], 1),
            grid_size_m=(self.bev_info['h_meters'], self.bev_info['w_meters'], 1),
            cmap={
                0: (0,0,0),
                1: (220,220,220),
                2: (255,127,80),
                3: (30,144,255),
            }
        )
        return output_bev, drivable_area_raster

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Additional labels for vehicles only.
        aux, visibility = self.get_dynamic_objects(sample, anns_vehicle)

        # Package the data.
        data = Sample(
            view=self.view.tolist(),
            bev=bev,
            aux=aux,
            visibility=visibility,
            **sample
        )

        if self.transform is not None:
            data = self.transform(data)

        return data
