import numpy as np
from scipy.spatial.transform import Rotation as R
from Lab1_FK_answers import part2_forward_kinematics


def find_distance(cur_pos, target_pose):
    dist_vec = cur_pos - target_pose
    return np.dot(dist_vec, dist_vec)


# assume the rotation is using the rotation represented by radian euler form
def find_JT(rotations, position):
    rows = np.shape(rotations)[0]
    JT = np.zeros((rows * 3, 3))

    parent_orientation = R.identity()

    # calculate jacobian matrix using geometry approach
    for i in range(0, rows):
        r = position[-1] - position[i]
        Rx = R.from_euler("X", [rotations[i][0]], degrees=True)
        Ry = R.from_euler("Y", [rotations[i][1]], degrees=True)
        Rz = R.from_euler("Z", [rotations[i][2]], degrees=True)
        ax = parent_orientation.apply([1, 0, 0])
        ay = (parent_orientation * Rx).apply([0, 1, 0])
        az = (parent_orientation * Rx * Ry).apply([0, 0, 1])

        JT[3 * i] = np.cross(ax, r)
        JT[3 * i + 1] = np.cross(ay, r)
        JT[3 * i + 2] = np.cross(az, r)

        parent_orientation *= Rx * Ry * Rz
    return JT


def find_gradient(path_rotation, path_offset, path_position, target_pose, meta_data):
    JT = find_JT(path_rotation, path_position)
    # jacobian inverse method
    damping_parameter = 1.0 / 10000.0
    grad = np.matmul(JT, np.linalg.inv(np.matmul(np.transpose(JT), JT) + damping_parameter * np.identity(3)))

    return np.matmul(grad, path_position[-1] - target_pose)


# Do forward kinematic to a chain of joint
# always assume the previous joint is the current joint's
# parent
def FK(rotations, offsets):
    orientation = np.zeros(np.shape(rotations))
    positions = np.zeros((np.shape(rotations)[0], 3))

    orientation[0] = rotations[0]
    parent_position = np.copy(offsets[0])
    positions[0] = np.copy(offsets[0])
    for i in range(1, np.shape(rotations)[0]):
        rotation = R.from_euler("XYZ", rotations[i], degrees=True)
        parent_orientation = R.from_euler("XYZ", orientation[i - 1], degrees=True)
        joint_orientation = parent_orientation * rotation
        orientation[i] = joint_orientation.as_euler('XYZ', degrees=True)
        positions[i] = positions[i - 1] + parent_orientation.apply(offsets[i])

    return positions, orientation


def f(rotations, offsets, target_position, initial_path_rotation):
    """
    calculate the loss function
    input:

    """

    damping_parameter = 1.0 / 10000.0
    positions, _, = FK(rotations, offsets)
    rotation_diff = np.reshape(rotations, (-1)) - np.reshape(initial_path_rotation, (-1))
    print(0.5 * find_distance(positions[-1], target_position))
    print(0.5 * damping_parameter * np.dot(rotation_diff, rotation_diff))
    return 0.5 * find_distance(positions[-1], target_position)\
           + 0.5 * damping_parameter * np.dot(rotation_diff, rotation_diff)


def line_search(z, dz, max_steps, offsets, target_position, initial_path_rotation):
    step = max_steps
    E0 = f(z, offsets, target_position, initial_path_rotation)
    z = np.reshape(z, (-1))
    while step > 1e-4:

        new_z = z + step * dz
        new_z = np.reshape(new_z, (-1, 3))
        clip_angle(new_z)
        E = f(new_z, offsets, target_position, initial_path_rotation)
        if E < E0:
            return step
        else:
            step /= 2.0
    return 0


def find_rotations(orientation, meta_data):
    """
    find the rotation of every joint
    input:
        orientation: joints global orientation
        meta_data: a data structure that has joint_initial_position, joint_parent and joint_name
    output:
        joint_rotation: the local rotation of every joint
    """

    joint_rotation = []
    for i in range(0, len(meta_data.joint_name)):
        name = meta_data.joint_name[i]
        parent_idx = meta_data.joint_parent[i]

        joint_orientation = R.from_quat(orientation[i])
        if name == "RootJoint":
            rotation = joint_orientation.as_euler('XYZ', degrees=True)
        elif name.find("_end") != -1:
            continue
        else:
            parent_orientation = R.from_quat(orientation[parent_idx])
            rotation = (parent_orientation.inv() * joint_orientation).as_euler('XYZ', degrees=True)

        joint_rotation.append(rotation.tolist())

    return np.array(joint_rotation)


def get_path_info(joint_orientation, joint_positions, meta_data):
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

    path, path_names, forward_path, backward_path = meta_data.get_path_from_root_to_end()
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
            rotation = orientation.as_euler('XYZ', degrees=True)
            position = offset
        else:
            parent_index = path[i - 1]
            position = joint_positions[joint_index]
            offset = meta_data.joint_initial_position[joint_index] - meta_data.joint_initial_position[parent_index]
            rotation = (parent_orientation.inv() * orientation).as_euler('XYZ', degrees=True)

        path_rotation.append(rotation)
        path_position.append(position)
        path_offset.append(offset)

        parent_orientation = orientation

    return np.array(path_rotation), np.array(path_offset), np.array(path_position)


def rotation_path_to_global(path_rotation, path_position, meta_data):
    """
    Input:
        path_rotation: numpy array nx3, rotation of a chain, joint i is the parent of joint i+1
        path_position: numpy array nx3, global position of every joint in the chain
        meta_data: useful data for the whole skeleton
    Output:
        global_rotation: numpy array nx3, rotation of the joint that is based on their actual parent in the skeleton
        root position: numpy array 3, global position of the root of the skeleton
    """

    global_rotation = np.zeros(np.shape(path_rotation))

    path, path_name, path1, path2 = meta_data.get_path_from_root_to_end()

    # the root of the chain is the root of skeleton, then global rotation is the
    # path_rotation
    if len(path2) == 1:
        return path_rotation, path_position[0]

    print("Have not implemented when root of chain is not root of the skeleton")
    exit(1)


def combine_rotation(path_rotation, joint_rotation, meta_data):
    """
    combine the new rotation in the chain and the rotation for other joint in the skeleton
    Input:
        path_rotation: numpy array (nx3), the rotation of the chain and each joint rotation is relative to its actual
        parent in the skeleton
        joint_rotation: numpy array (nx3), old joint rotation
        meta_data: useful data for the whole skeleton

    output:
        new_joint_rotation: numpy array (nx3), new joint rotation
    """
    path, path_name, _, _ = meta_data.get_path_from_root_to_end()

    new_joint_rotation = np.copy(joint_rotation)

    for i in range(0, len(path)):
        if path_name[i].find("_end") != -1:
            continue
        assert(0 <= path[i] < np.shape(new_joint_rotation)[0])
        new_joint_rotation[path[i]] = path_rotation[i]

    return new_joint_rotation


def get_offset(meta_data):
    """
    get the joint offset of the skeleton
    Input:
        meta_data: useful data structure for skeleton
    Output:
        joint_offset: numpy array (nx3), the offset of the joint
    """
    joint_offset = []
    joint_name = meta_data.joint_name
    joint_parent = meta_data.joint_parent
    joint_initial_position = meta_data.joint_initial_position

    for i in range(0, len(joint_name)):
        if joint_parent[i] == -1:
            joint_offset.append([0.0, 0.0, 0.0])
        else:
            parent_idx = joint_parent[i]
            offset = joint_initial_position[i] - joint_initial_position[parent_idx]
            joint_offset.append(offset.tolist())
    return np.array(joint_offset)


def clip_angle(rotation):
    """
    clip the rotation angle to [-180, 180]
    Input:
        rotation: rotation of each joint
    output:
        result: rotation of each joint after clipping
    """
    result = np.copy(rotation)
    for i in range(0, np.shape(result)[0]):
        result[i] = (R.from_euler("XYZ", result[i], degrees=True)).as_euler("XYZ", degrees=True)
    return result


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

    MAX_ITERATION = 200
    TOLERANCE = 0.01

    # find the rotation of every joint
    joint_rotation = find_rotations(joint_orientations, meta_data)

    # filter out the relative current rotation based on the chain
    path_rotation, path_offset, path_position = get_path_info(joint_orientations, joint_positions, meta_data)
    initial_path_rotation = np.copy(path_rotation)
    # optimization
    iter = 0
    print(find_distance(path_position[-1], target_pose))
    print(f(path_rotation, path_offset, target_pose, initial_path_rotation))
    while iter < MAX_ITERATION and find_distance(path_position[-1], target_pose) > TOLERANCE:
        gradient = find_gradient(path_rotation, path_offset, path_position, target_pose, meta_data)

        sigma = line_search(path_rotation, -gradient, 1000, path_offset, target_pose, initial_path_rotation)
        path_rotation_1D = np.reshape(path_rotation, (-1))
        path_rotation_1D = path_rotation_1D - sigma * gradient
        path_rotation = np.reshape(path_rotation_1D, (-1, 3))
        path_rotation = clip_angle(path_rotation)


        path_position, _,= FK(path_rotation, path_offset)
        print(find_distance(path_position[-1], target_pose))
        iter += 1


    # put the chain rotation back to the rotation in the reference of root
    path_rotation, root_position = rotation_path_to_global(path_rotation, path_position, meta_data)

    print("rotation path to global")
    print(path_rotation)

    print(root_position)
    # combine every rotation together
    new_joint_rotations = combine_rotation(path_rotation, joint_rotation, meta_data)
    print("new rotation")

    path, _, _, _ = meta_data.get_path_from_root_to_end()
    print(path)
    print(joint_rotation)
    print(path_rotation)
    print(new_joint_rotations)

    motion_data = np.concatenate((root_position, new_joint_rotations), axis=None).tolist()
    motion_data = [motion_data]
    motion_data = np.array(motion_data)
    print("motion data")
    print(motion_data)

    joint_offset = get_offset(meta_data)
    print("joint offset", joint_offset)
    # calculate the joint position and joint orientation
    joint_positions, joint_orientations = part2_forward_kinematics(meta_data.joint_name, meta_data.joint_parent,
                                                                   joint_offset, motion_data, 0)
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