from utils.r3D_semantic_dataset import R3DSemanticDataset
import numpy as np
import pandas as pd
from pyntcloud import PyntCloud
import matplotlib.pyplot as plt
import open3d as o3d
from scipy.spatial.transform import Rotation as R

from pyntcloud.plot.common import get_colors

from matplotlib.patches import Circle
import mpl_toolkits.mplot3d.art3d as art3d
import matplotlib.cm as cm
from mpl_toolkits import mplot3d

from utils.visualizer.meshes.load_mesh import get_gripper_loader
import PIL
from PIL import Image


POSE_AXES_LEN_SCALE = 0.1
AMBIENT_ROTATION_SPEED = 0.5
AMBIENT_ROTATION_MAGNITUDE = 10.0
AUG = 0.6


load_gripper = get_gripper_loader()


def action_tensor_to_matrix(action_tensor):
    affine = np.eye(4)
    r = R.from_quat(action_tensor[:4])
    affine[:3, :3] = r.as_matrix()
    affine[:3, -1] = action_tensor[4:]

    return affine


def get_camera_view_param(ctr, affine):
    # initialize view using affine (directly plugging in as extrensic matrix doesnt seem to work well)
    ctr.set_lookat(affine[:3, 3])
    ctr.set_up(-np.squeeze(affine[:4, 0])[:3])
    ctr.set_front(np.squeeze(affine[:4, 2])[:3])
    # get camera parameters of the current view
    params = ctr.convert_to_pinhole_camera_parameters()

    # update extrinsic matrix of the camera parameters using the r3d affine
    ext_inv = np.linalg.inv(params.extrinsic)
    ext_inv[:4, 3] = affine[:4, 3]
    params.extrinsic = np.linalg.inv(ext_inv)
    return params


def stl_to_o3d_mesh(stl_mesh):
    vertices = np.array(stl_mesh.vectors.reshape(-1, 3))
    triangles = np.arange(vertices.shape[0]).reshape(-1, 3)
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(triangles)
    # change o3d mesh color to blue
    o3d_mesh.paint_uniform_color([0.0, 0.0, 1.0])
    return o3d_mesh


def set_proper_aspect_ratio(ax):
    extents = np.array([getattr(ax, "get_{}lim".format(dim))() for dim in "xyz"])
    sz = extents[:, 1] - extents[:, 0]
    centers = np.mean(extents, axis=1)
    maxsize = max(abs(sz))
    r = maxsize / 2
    for ctr, dim in zip(centers, "xyz"):
        getattr(ax, "set_{}lim".format(dim))(ctr - r, ctr + r)


def plot_affine(affine, ax, gripper=1, azim=-60, plot=True, path=None, annot=None):
    position = affine[:3, 3]
    pose_axes = affine[:3, :3].T * POSE_AXES_LEN_SCALE  # why is he taking transpose?

    pose_axes_endpoints = pose_axes + position
    x, y, z = position
    _ = ax.scatter(x, y, z, color="b")
    #     _ = ax.plot([x, x], [y, y], [z, 0], alpha=0.5, color='k')
    #     _ = ax.scatter(x, y, 0, alpha=0.5, color='k')

    if annot is not None:
        ax.text(x, y, z, annot)
    # print(-affine[:3, 1])
    r = R.from_matrix(affine[:3, :3])

    gripper_mesh = load_gripper((x, y, z), r, gripper, scale=0.75)
    ax.add_collection3d(mplot3d.art3d.Poly3DCollection(gripper_mesh.vectors))
    scale = AUG * gripper_mesh.points.flatten()

    q = r.as_quat()
    euler = r.as_euler("xyz")  # roll pitch yaw (in vertical)
    roll, pitch, yaw = np.rad2deg(euler)

    axes_cmap = "rgb"
    for i in range(3):  # draw pose axes
        endpoint = pose_axes_endpoints[i, :]
        ax.plot(*zip(position, endpoint), color=axes_cmap[i], alpha=0.5)


def action_tensor_to_matrix(action_tensor):
    affine = np.eye(4)
    r = R.from_quat(action_tensor[:4])
    affine[:3, :3] = r.as_matrix()
    affine[:3, -1] = action_tensor[4:]

    return affine


def get_o3d_mesh_plotter(cloud, **kwargs):
    vis = o3d.visualization.Visualizer()
    vis.create_window(visible=False)

    vis.add_geometry(cloud)
    vis.update_geometry(cloud)
    affine = action_tensor_to_matrix(kwargs["pose"])
    init_affine = np.eye(4)
    if "init_pose" in kwargs:
        init_affine = action_tensor_to_matrix(kwargs["init_pose"])

    ctr = vis.get_view_control()
    c_param = get_camera_view_param(ctr, init_affine @ affine)

    def mesh_plotter(pose, gripper=1, dx=30, dy=160, zoom=0.7):
        affine_corr = init_affine @ action_tensor_to_matrix(pose)
        position = affine_corr[:3, 3]
        x, y, z = position
        r = R.from_matrix(affine_corr[:3, :3])
        gripper_mesh = stl_to_o3d_mesh(load_gripper((x, y, z), r, gripper, scale=0.75))

        vis.add_geometry(gripper_mesh)
        vis.update_geometry(gripper_mesh)

        # c_param = get_camera_view_param(affine_corr)
        # ctr.change_field_of_view(step=90.0)
        ctr.convert_from_pinhole_camera_parameters(c_param, allow_arbitrary=True)
        ctr.rotate(dx, dy)
        ctr.set_zoom(zoom)
        vis.poll_events()
        vis.update_renderer()
        # vis.capture_screen_image(
        #     "/scratch/ar7420/VINN/imitation-in-homes/output_image_test.png"
        # )
        # print(vis.capture_screen_float_buffer(True))
        np_img = np.asarray(vis.capture_screen_float_buffer(True))
        np_img = (255.0 * np_img).astype(np.uint8)
        # change BGR to RGB
        np_img = np_img[:, :, ::-1]
        vis.remove_geometry(gripper_mesh)
        return np_img

    return mesh_plotter


def plot_with_matplotlib(cloud, **kwargs):
    colors = get_colors(cloud, kwargs["use_as_color"], kwargs["cmap"])
    #     print(colors)
    ptp = cloud.xyz.ptp()

    plt.figure(figsize=(10, 10))
    ax = plt.axes(projection="3d")
    ax.view_init(
        elev=kwargs.get("elev", 0.0),
        azim=kwargs.get("azim", 90.0),
        roll=kwargs.get("roll", 0.0),
    )

    ax.scatter(
        cloud.xyz[:, 0],
        cloud.xyz[:, 1],
        cloud.xyz[:, 2],
        marker="D",
        facecolors=colors / 255,
        alpha=0.25,
        zdir="z",
        depthshade=True,
        s=kwargs["initial_point_size"] or ptp / 10,
    )

    if "text" in kwargs:
        for i, (points, text) in enumerate(kwargs["text"]):
            ax.text(points[0], points[1], points[2], text)

    if "title" in kwargs:
        ax.set_title(kwargs["title"])

    ax.set_xlabel("X axis")
    ax.set_ylabel("Y axis")
    ax.set_zlabel("Z axis")
    #     for i in range(0, 30,30):
    #         print(i)
    if "pose" in kwargs:
        affine = action_tensor_to_matrix(kwargs["pose"])
        if "init_pose" in kwargs:
            init_affine = action_tensor_to_matrix(kwargs["init_pose"])
            affine = init_affine @ affine
        plot_affine(
            affine,
            ax,
            gripper=kwargs.get("gripper", 1),
            annot=kwargs.get("annot", None),
        )

    set_proper_aspect_ratio(ax)
    plt.tight_layout()
    fig1 = plt.gcf()

    if "plot" not in kwargs or kwargs["plot"]:
        plt.show()

    fig1.canvas.draw()
    image = PIL.Image.frombytes(
        "RGB", fig1.canvas.get_width_height(), fig1.canvas.tostring_rgb()
    )
    # convert to cv2 image
    image = np.array(image)
    # save image
    image = image[:, :, ::-1]
    # wait

    # close the figure
    plt.close(fig1)
    return image
