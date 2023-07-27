import liblzfse
import numpy as np
import cv2
import PIL
import matplotlib.pyplot as plt
import open3d as o3d
from quaternion import quaternion, as_rotation_matrix

from PIL import Image
import json
import tqdm
from torch.utils.data import Dataset
from typing import List
import json
import os


def load_image(filepath):
    return np.asarray(Image.open(filepath))


def load_depth(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.float32)

    # depth_img = depth_img.reshape((960, 720))  # For a FaceID camera 3D Video
    depth_img = depth_img.reshape((256, 192))  # For a LiDAR 3D Video

    return depth_img


def load_conf(filepath):
    with open(filepath, "rb") as depth_fh:
        raw_bytes = depth_fh.read()
        decompressed_bytes = liblzfse.decompress(raw_bytes)
        depth_img = np.frombuffer(decompressed_bytes, dtype=np.uint8)
    return depth_img


class R3DSemanticDataset(Dataset):
    def __init__(
        self,
        path: str,
        custom_classes: List[str] = None,
        max_frames_to_load: int = None,
    ):
        self._path = path
        self._max_frames_to_load = max_frames_to_load
        self._reshaped_depth = []
        self._reshaped_conf = []
        self._depth_images = []
        self._rgb_images = []
        self._confidences = []

        self._metadata = self._read_metadata(path, custom_classes)
        self.global_xyzs = []
        self.global_pcds = []
        self._load_data()
        self._reshape_all_depth_and_conf()
        self.calculate_all_global_xyzs()

    def _read_metadata(self, path, custom_classes):
        with open(f"{path}/metadata", "r") as f:
            metadata_dict = json.load(f)

        # Now figure out the details from the metadata dict.
        self.rgb_width = metadata_dict["w"]
        self.rgb_height = metadata_dict["h"]
        self.image_size = (self.rgb_width, self.rgb_height)

        self.poses = np.array(metadata_dict["poses"])
        self.camera_matrix = np.array(metadata_dict["K"]).reshape(3, 3).T

        self.fps = metadata_dict["fps"]

        self.total_images = (
            len(self.poses)
            if self._max_frames_to_load is None
            else self._max_frames_to_load
        )

        self.init_pose = np.array(metadata_dict["initPose"])

        self._id_to_name = {
            i: x for (i, x) in enumerate(custom_classes)
        }  # There are no predefined IDs and names.

    def _load_data(self):
        assert self.fps  # Make sure metadata is read correctly first.
        for i in tqdm.trange(self.total_images):
            # Read up the RGB and depth images first.

            # check if the folder exists first
            if os.path.exists(f"{self._path}/rgbd"):
                rgb_filepath = f"{self._path}/rgbd/{i}.jpg"
                depth_filepath = f"{self._path}/rgbd/{i}.depth"
                conf_filepath = f"{self._path}/rgbd/{i}.conf"
            else:
                # add zeros to the file name to make it 4 digits
                rgb_filepath = f"{self._path}/images/{i:04d}.jpg"
                depth_filepath = f"{self._path}/depths/{i:04d}.depth"
                conf_filepath = f"{self._path}/confs/{i:04d}.conf"
            depth_img = load_depth(depth_filepath)
            rgb_img = load_image(rgb_filepath)
            confidence = load_conf(conf_filepath).reshape(256, 192)

            # Now, convert depth image to real world XYZ pointcloud.
            self._depth_images.append(depth_img)
            self._rgb_images.append(rgb_img)
            self._confidences.append(confidence)

    def _reshape_all_depth_and_conf(self):
        for index in tqdm.trange(self.total_images):
            depth_image = self._depth_images[index]
            # Upscale depth image.
            pil_img = Image.fromarray(depth_image)
            reshaped_img = pil_img.resize((self.rgb_width, self.rgb_height))
            # all_rgbs.append()
            reshaped_img = np.asarray(reshaped_img)
            self._reshaped_depth.append(reshaped_img)

            # Upscale confidence as well
            confidence = self._confidences[index]
            # Upscale depth image.
            conf_img = Image.fromarray(confidence)
            reshaped_conf = conf_img.resize((self.rgb_width, self.rgb_height))
            # all_rgbs.append()
            reshaped_conf = np.asarray(reshaped_conf)
            self._reshaped_conf.append(reshaped_conf)

    def get_global_xyz(self, index, depth_scale=1000.0, only_confident=True):
        reshaped_img = np.copy(self._reshaped_depth[index])
        # If only confident, replace not confident points with nans
        if only_confident:
            reshaped_img[self._reshaped_conf[index] != 2] = np.nan
        # switch first and second axis in reshaped_img

        depth_o3d = o3d.geometry.Image(
            np.ascontiguousarray(depth_scale * reshaped_img).astype(np.float32)
        )

        if reshaped_img.shape[:] != self._rgb_images[index].shape[:2]:
            rotated_img = np.rot90(self._rgb_images[index], 1)
            rgb_o3d = o3d.geometry.Image(
                np.ascontiguousarray(rotated_img).astype(np.uint8)
            )

        else:
            rgb_o3d = o3d.geometry.Image(
                np.ascontiguousarray(self._rgb_images[index]).astype(np.uint8)
            )
        # print number of nan values in depth image
        # print(self._rgb_images[index].shape)

        # print(np.count_nonzero(np.isnan(reshaped_img)), reshaped_img.shape)
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb_o3d, depth_o3d, convert_rgb_to_intensity=False
        )
        # ratio = 2 56 / 960
        ratio = 1

        camera_intrinsics = o3d.camera.PinholeCameraIntrinsic(
            width=int(self.rgb_width * ratio),
            height=int(self.rgb_height * ratio),
            fx=self.camera_matrix[0, 0] * ratio,
            fy=self.camera_matrix[1, 1] * ratio,
            cx=self.camera_matrix[0, 2] * ratio,
            cy=self.camera_matrix[1, 2] * ratio,
        )
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, camera_intrinsics
        )
        # Flip the pcd
        pcd.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        extrinsic_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.poses[index]
        extrinsic_matrix[:3, :3] = quaternion.as_rotation_matrix(quaternion(qw, qx, qy, qz))
        extrinsic_matrix[:3, -1] = [px, py, pz]
        pcd.transform(extrinsic_matrix)

        # Now transform everything by init pose.
        # self.init_pose = np.array(self.poses[5])
        init_matrix = np.eye(4)
        qx, qy, qz, qw, px, py, pz = self.init_pose
        init_matrix[:3, :3] = quaternion.as_rotation_matrix(quaternion(qw, qx, qy, qz))
        init_matrix[:3, -1] = [px, py, pz]
        pcd.transform(init_matrix)

        return pcd

    def calculate_all_global_xyzs(self, only_confident=True):
        if len(self.global_xyzs):
            return self.global_xyzs, self.global_pcds
        for i in tqdm.trange(self.total_images):
            global_xyz_pcd = self.get_global_xyz(i, only_confident=only_confident)
            global_xyz = np.asarray(global_xyz_pcd.points)
            self.global_xyzs.append(global_xyz)
            self.global_pcds.append(global_xyz_pcd)
        return self.global_xyzs, self.global_pcds

    def __len__(self):
        return len(self.poses)

    def __getitem__(self, idx):
        result = {
            "xyz_position": self.global_xyzs[idx],
            "rgb": self._rgb_images[idx],
            "depth": self._reshaped_depth[idx],
            "conf": self._reshaped_conf[idx],
        }
        return result
