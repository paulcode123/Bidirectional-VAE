from PIL import Image
import glob
import numpy as np
import pickle
import cv2
from utils.visualizer.plot_3d import plot_with_matplotlib, get_o3d_mesh_plotter
import argparse
import os
from utils.action_transforms import *
from utils.r3D_semantic_dataset import R3DSemanticDataset
import torchvision
import torch
import open3d as o3d
import pandas as pd
import matplotlib.pyplot as plt
from pyntcloud import PyntCloud
import json
from tqdm import tqdm


############# Helper functions for building videos #####################


def make_video(frames, path, save_path, postfix="", downsample_by=2, FPS=30):
    basename = os.path.basename(path)
    # combine all frames into a mp4 video using opencv
    height, width, layers = frames[0].shape
    size = (width, height)
    tot_frames = len(frames) // downsample_by
    time = len(frames) // FPS
    fps = tot_frames // time

    multiplier = len(frames) / tot_frames

    # sample using multiplier

    frames = [
        cv2.cvtColor(frames[int(i * multiplier)], cv2.COLOR_BGR2RGB)
        for i in range(tot_frames)
    ]
    # convert to numpy array
    frames = [torch.Tensor(np.array(frame)) for frame in frames]

    # use torchvision to save the video
    torchvision.io.write_video(
        save_path + "/" + basename + postfix + ".mp4", torch.stack(frames), fps
    )


def get_concat_v(im1, im2):
    dst = Image.new("RGB", (im1.width, im1.height + im2.height))
    dst.paste(im1, (0, 0))
    dst.paste(im2, (0, im1.height))
    return dst


def combine_images(frame, plot1, plot2):
    # vertically stack plot1 and plot2 images
    # resize plot2 to match shape of plot1 for horizonta stacking, keep aspect ratio
    aspect_ratio = plot2.shape[1] / plot2.shape[0]
    plot2 = cv2.resize(plot2, (int(plot1.shape[0] * aspect_ratio), plot1.shape[0]))

    plot_v1 = np.hstack((plot1, plot2))
    # resize frame to match shape of plot1

    frame = cv2.resize(frame, (int(plot1.shape[1] // 2.5), int(plot1.shape[0] // 2.5)))
    # add black border on left and right to frame to match shape of plot_v1 for vertical stacking
    diff_w = plot_v1.shape[1] - frame.shape[1]
    frame = cv2.copyMakeBorder(
        frame,
        0,
        0,
        diff_w // 2,
        diff_w // 2,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    # print(frame.shape, plot_v1.shape)
    # horizontally stack plot_v1 and plot_v2
    combined = np.vstack((plot_v1, frame))
    return combined


def get_pyntcld(cloud_dataset):
    pcd_lite = cloud_dataset.global_pcds[0]
    pcd_lite.colors = o3d.utility.Vector3dVector(np.asarray(pcd_lite.colors) * 255)
    pts_result = np.concatenate(
        (np.asarray(pcd_lite.points), np.asarray(pcd_lite.colors)), axis=-1
    )

    df = pd.DataFrame(
        # same arguments that you are passing to visualize_pcl
        data=pts_result,
        columns=["x", "y", "z", "red", "green", "blue"],
    )
    return PyntCloud(df)


############# functions to build 3d video of point cloud with gripper overlay #####################


def build_o3dvis_for_trajectory(
    path, save_path, add_plot=True, reduce_size_by=2, reduce_freq_by=2
):
    cloud_dataset = R3DSemanticDataset(path, [], max_frames_to_load=1)

    o3d_mesh_plotter = get_o3d_mesh_plotter(
        cloud_dataset.global_pcds[0],
        pose=cloud_dataset.poses[0],
        init_pose=cloud_dataset.init_pose,
    )

    images = []
    # check if labels.json exists
    if os.path.exists(
        path + "/labels.json"
    ):  # loading when the data is already processed
        with open(path + "/labels.json") as f:
            labels = json.load(f)

        keys = list(labels.keys())
    else:
        keys = [str(i) for i in range(len(cloud_dataset.poses))]
        # zfill keys to have 4 digits and add .jpg extension
        keys = [k.zfill(4) + ".jpg" for k in keys]
        labels = {k: {"gripper": 1} for k in keys}

    for i, k in tqdm(enumerate(keys)):
        if i % reduce_freq_by != 0:
            continue

        # calculate the ratio of the current frame to the total number of frames
        ratio = i / len(keys)

        frame = cv2.imread(path + "/images/" + k)
        if reduce_size_by:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] / reduce_size_by),
                    int(frame.shape[0] / reduce_size_by),
                ),
            )
        if add_plot:
            plot1 = o3d_mesh_plotter(
                cloud_dataset.poses[i],
                labels[k]["gripper"],
                dx=int(30 + 45 * ratio),
                dy=int(150 - 45 * ratio),
                zoom=0.65 - (0.15 * ratio),
            )
            plot2 = o3d_mesh_plotter(
                cloud_dataset.poses[i],
                labels[k]["gripper"],
                dx=int(-40 - 45 * ratio),
                dy=int(-10 + 45 * ratio),
                zoom=0.65 - (0.15 * ratio),
            )

            frame = combine_images(frame, plot1, plot2)
        images.append(frame)

    # save video
    make_video(images, path, save_path, downsample_by=1, FPS=30 / reduce_freq_by)


def build_3dvis_for_trajectory(
    path, save_path, add_plot=True, reduce_size_by=2, reduce_freq_by=2
):
    print("building vis for trajectory at: ", path)
    cloud_dataset = R3DSemanticDataset(path, [])
    cloud = get_pyntcld(cloud_dataset)
    # load labels from json file
    images = []
    with open(path + "/labels.json") as f:
        labels = json.load(f)
    keys = list(labels.keys())
    for i, k in tqdm(enumerate(keys)):
        if i % reduce_freq_by != 0:
            continue
        frame = cv2.imread(path + "/images/" + k)
        if reduce_size_by:
            frame = cv2.resize(
                frame,
                (
                    int(frame.shape[1] / reduce_size_by),
                    int(frame.shape[0] / reduce_size_by),
                ),
            )
        combined = frame
        if add_plot:
            # print(type(relative_poses))

            plot1 = plot_with_matplotlib(
                cloud,
                width=800,
                height=500,
                background="black",
                mesh=False,
                use_as_color=["red", "green", "blue"],
                initial_point_size=None,
                cmap="hsv",
                polylines=None,
                linewidth=5,
                return_scene=False,
                output_name="pyntcloud_plot",
                elev=120.0 + (i * 20) / len(keys),
                azim=-130.0 - (i * 20) / len(keys),
                roll=-50.0,
                pose=cloud_dataset.poses[i],
                init_pose=cloud_dataset.init_pose,
                annot=str(i),
                gripper=labels[k]["gripper"],
                plot=False,
            )

            plot2 = plot_with_matplotlib(
                cloud,
                width=800,
                height=500,
                background="black",
                mesh=False,
                use_as_color=["red", "green", "blue"],
                initial_point_size=None,
                cmap="hsv",
                polylines=None,
                linewidth=5,
                return_scene=False,
                output_name="pyntcloud_plot",
                elev=25.0 - (i * 20) / len(keys),
                azim=120.0 + (i * 20) / len(keys),
                roll=200.0,
                pose=cloud_dataset.poses[i],
                init_pose=cloud_dataset.init_pose,
                annot=str(i),
                gripper=labels[k]["gripper"],
                plot=False,
            )

            # cv2.imshow('plot2', plot2)
            combined = combine_images(frame, plot1, plot2)
            # convert cv2 image to PIL image
        images.append(combined)
    make_video(images, path, save_path, downsample_by=1, FPS=30 / reduce_freq_by)
