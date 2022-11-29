import numpy as np
from scipy.spatial.transform import Rotation as R

# assume the rotation is using the rotation represented by radian euler form
def find_JT(rotations, position, target):
    rows = np.shape(rotations)[0]
    JT = np.zeros((rotations.shape()[0] * 3, 3))

    parent_orientation = R.identity()

    # calculate jacobian matrix using geometry approach
    for i in range(0, rows):
        r = target - position[i]
        Rx = R.from_euler("X", [rotations[i][0]])
        Ry = R.from_euler("Y", [rotations[i][1]])
        Rz = R.from_euler("Z", [rotations[i][2]])
        ax = parent_orientation.apply([1, 0, 0])
        ay = (parent_orientation * Rx).apply([0, 1, 0])
        az = (parent_orientation * Rx * Ry).apply([0, 0, 1])

        JT[3 * i] = np.cross(ax, r)
        JT[3 * i + 1] = np.cross(ay, r)
        JT[3 * i + 2] = np.cross(az, r)

        parent_orientation *= Rx * Ry * Rz

# Do forward kinematic to a chain of joint
# always assume the previous joint is the current joint's
# parent
def FK(rotations, offsets):
    orientation = np.zeros(np.shape(rotations))
    positions = np.zeros((np.shape(rotations)[0], 3))

    parent_orientation = R.identity()
    parent_position = np.array([0, 0, 0])
    for i in range(0, np.shape(rotations)[0]):
        rotation = R.from_euler("XYZ", rotations[i])
        orientation = parent_orientation * rotation
        positions = parent_position + parent_orientation.apply(offsets[i])

        parent_orientation = orientation

    return positions, orientation


def find_rotations(orientation, meta_data):
    """
    find the rotation of every joint
    input:
        orientation: joints global orientation
        meta_data: a data structure that has joint_initial_position, joint_parent and joint_name
    output:
        joint_rotation: the local rotation of every joint
    """

def get_path_info(joint_orientation, joint_positions, meta_data, path, forward_path, backward_path):
    """
    filter the rotation, position and offset of the joint on the path/chain only
    it will treat the beginning of the chain as a root and end of the chain as end joint
    input:
        joint_orientation: every joint's global orientation in quaternion form
        joint_position: every joint's global position
        meta_data: a data structure that has joint_initial_position, joint_parent and joint_name
        path: a list of index of joint that are in the chain
        forward_path: second part of the path that start at root joint
        backward_path: first part of the path that ends at root joint
    output:
        path_rotation: rotation of each joint relative to their parent on the chain, in Euler radian form
        path_offset: offset of each joint relative to their parent on the chain, the offset of first joint is relative
        to (0, 0, 0) to global coordinate
        path_position: global position of the joint in path
    """

    path_rotation = []
    path_offset = []
    path_position = []

    parent_orientation = R.identity()
    for i in range(0, len(path)):

        joint_index = path[i]
        # ignore root joint if it is not at the start of the chain
        if i != 0 and joint_index == 0:
            continue

        orientation = R.from_quat(joint_orientation[joint_index])
        if i == 0:
            offset = joint_positions[joint_index]
            rotation = orientation.as_euler('XYZ')
            position = offset
        else:
            parent_index = path[i - 1]
            position = joint_positions[joint_index]
            offset = meta_data.joint_initial_position[joint_index] - meta_data.joint_initial_position[parent_index]
            rotation = (parent_orientation.inv() * orientation).as_euler('XYZ')

        path_rotation.append(rotation)
        path_position.append(position)
        path_offset.append(offset)

        parent_orientation = orientation

    return np.array(path_rotation), np.array(path_offset), np.array(path_position)


def part1_inverse_kinematics(meta_data, joint_positions, joint_orientations, target_pose):
    """
    完成函数，计算逆运动学
    输入: 
        meta_data: 为了方便，将一些固定信息进行了打包，见上面的meta_data类
        joint_positions: 当前的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 当前的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
        target_pose: 目标位置，是一个numpy数组，shape为(3,)
    输出:
        经过IK后的姿态
        joint_positions: 计算得到的关节位置，是一个numpy数组，shape为(M, 3)，M为关节数
        joint_orientations: 计算得到的关节朝向，是一个numpy数组，shape为(M, 4)，M为关节数
    """
    path, path_names, forward_path, backward_path = meta_data.get_path_from_root_to_end()
    forward_path = list(reversed(forward_path))
    """
    # initial stepping
    alpha = 1

    # tolerance for the difference between the target position and the result position
    tolerance = 0.01
    max_iteration = 20

    # current rotation for every joint
    joint_rotations = find_rotation(joint_orientations, meta_data)

    # current location and rotation in the path, rotation is in euler radian form
    path_rotation, path_position, path_offset


    iter = 0
    while iter < max_iteration and np.linalg.norm(path_position[-1] - target_pose) > tolerance:
        jacobian_transpose = find_JT(path_rotation, path_position, target_pose)
        old_distance = np.linalg.norm(path_position[-1] - target_pose)

        is_decreasing = False

        while not is_decreasing:
            path_rotation_1D = np.reshape(path_rotation, (-1))
            path_rotation_1D = path_rotation_1D - alpha * jacobian_transpose * (path_position[-1] - target_pose)
            path_rotation = np.reshape(path_rotation_1D, (-1, 3))

            path_position, path_orientations = FK(path_rotation, path_offset)
            if np.linalg.norm(path_position[-1] - target_pose) >= old_distance:
                alpha /= 2.0
            else:
                is_decreasing = True

    """
    
    return joint_positions, joint_orientations

def part2_inverse_kinematics(meta_data, joint_positions, joint_orientations, relative_x, relative_z, target_height):
    """
    输入lWrist相对于RootJoint前进方向的xz偏移，以及目标高度，IK以外的部分与bvh一致
    """


    return joint_positions, joint_orientations

def bonus_inverse_kinematics(meta_data, joint_positions, joint_orientations, left_target_pose, right_target_pose):
    """
    输入左手和右手的目标位置，固定左脚，完成函数，计算逆运动学
    """
    
    return joint_positions, joint_orientations