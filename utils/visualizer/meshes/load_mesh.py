from stl import mesh
from stl.base import RemoveDuplicates
from mpl_toolkits import mplot3d
from matplotlib import pyplot
import math
from scipy.spatial.transform import Rotation as R

# import vtkplotlib as vpl
import numpy as np
import pyfqmr
import os

mesh_simplifier = pyfqmr.Simplify()


def get_gripper_loader():
    current_dir = os.path.dirname(os.path.realpath(__file__))
    left_gripper_finger = mesh.Mesh.from_file(
        current_dir + "/simplify_gripper_combined_left.stl",
        remove_empty_areas=True,
        remove_duplicate_polygons=RemoveDuplicates.SINGLE,
    )
    right_gripper_finger = mesh.Mesh.from_file(
        current_dir + "/simplify_gripper_combined_right.stl",
        remove_empty_areas=True,
        remove_duplicate_polygons=RemoveDuplicates.SINGLE,
    )
    gripper_middle = mesh.Mesh.from_file(
        current_dir + "/simplify_link_straight_gripper.stl",
        remove_empty_areas=True,
        remove_duplicate_polygons=RemoveDuplicates.SINGLE,
    )

    def load_gripper_fingers(coords, rotation, gripper, scale=1.0):
        # gripper: value from 0 to 1, 0 is closed, 1 is open
        # coords: x,y,z
        # rotation: scipy.spatial.transform.Rotation object

        left_gripper_finger_cp = mesh.Mesh(left_gripper_finger.data.copy())
        right_gripper_finger_cp = mesh.Mesh(right_gripper_finger.data.copy())
        gripper_middle_cp = mesh.Mesh(gripper_middle.data.copy())

        # rotate gripper fingers based on gripper value (15 degrees open, 0 degrees closed)
        left_gripper_finger_cp.rotate([0.0, 0.0, 1.0], math.radians(-15 * gripper))
        right_gripper_finger_cp.rotate([0.0, 0.0, 1.0], math.radians(15 * gripper))

        # translate gripper fingers to correct position
        left_gripper_finger_cp.translate([-0.005, 0.0, 0.0])
        right_gripper_finger_cp.translate([+0.005, 0.0, 0.0])

        # put the gripper_middle in the correct position
        r = R.from_euler("zx", [180, -90], degrees=True)
        rot_vec = r.as_rotvec()
        gripper_middle_cp.rotate(rot_vec, -np.linalg.norm(rot_vec))
        gripper_middle_cp.translate([0.0, -0.1, 0.0])

        # combine all meshes into one mesh
        combined_gripper = mesh.Mesh(
            np.concatenate(
                [
                    left_gripper_finger_cp.data,
                    right_gripper_finger_cp.data,
                    gripper_middle_cp.data,
                ]
            )
        )

        # scale the combined mesh
        combined_gripper.vectors *= scale

        # rotate by 90 along x
        combined_gripper.rotate(
            [1.0, 0.0, 0.0], math.radians(90)
        )  # rotate by 90 degrees along x
        # rotate along z by 90
        combined_gripper.rotate(
            [0.0, 0.0, 1.0], -math.radians(90)
        )  # rotate by 90 degrees along x

        # rotate and translate the combined mesh
        combined_gripper.rotate(
            rotation.as_rotvec(), -np.linalg.norm(rotation.as_rotvec())
        )
        combined_gripper.translate(coords)
        return combined_gripper

    return load_gripper_fingers


# Create a new plot


# main
if __name__ == "__main__":
    figure = pyplot.figure()
    axes = figure.add_subplot(projection="3d")

    load_gripper_fingers = get_gripper_loader()

    gripper = load_gripper_fingers(
        [0.25, 0.2, 0.1], R.from_euler("xyz", [-45, 45, 0], degrees=True), 0.3
    )

    scale = gripper.points.flatten()
    axes.add_collection3d(mplot3d.art3d.Poly3DCollection(gripper.vectors))

    # Auto scale to the mesh size
    # scale = your_mesh.points.flatten()
    aug = 1
    axes.auto_scale_xyz(aug * scale, aug * scale, aug * scale)
    # show axis
    axes.set_xlabel("X")
    axes.set_ylabel("Y")
    axes.set_zlabel("Z")
    # Show the plot to the screen
    pyplot.show()
