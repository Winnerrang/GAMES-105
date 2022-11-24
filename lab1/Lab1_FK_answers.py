import numpy as np
import math
from scipy.spatial.transform import Rotation as R

def load_motion_data(bvh_file_path):
    """part2 辅助函数，读取bvh文件"""
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('Frame Time'):
                break
        motion_data = []
        for line in lines[i+1:]:
            data = [float(x) for x in line.split()]
            if len(data) == 0:
                break
            motion_data.append(np.array(data).reshape(1,-1))
        motion_data = np.concatenate(motion_data, axis=0)
    return motion_data



def part1_calculate_T_pose(bvh_file_path):
    """请填写以下内容
    输入： bvh 文件路径
    输出:
        joint_name: List[str]，字符串列表，包含着所有关节的名字
        joint_parent: List[int]，整数列表，包含着所有关节的父关节的索引,根节点的父关节索引为-1
        joint_offset: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的偏移量

    Tips:
        joint_name顺序应该和bvh一致
    """

    joint_name = []
    joint_parent = []
    joint_offset = []
    stack = []
    dimension = 3
    with open(bvh_file_path, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if lines[i].startswith('ROOT'):
                break

        joint_name.append(lines[i].split(" ")[1].strip())
        joint_parent.append(-1);
        stack.append(0)
        i += 2

        while len(stack) != 0:

            line = lines[i].strip()
            lineList = list(filter(None, line.split(' ')))

            if lineList[0] == "JOINT" or lineList[0] == "End":

                # end joint name should be its parent joint's name + "_end"
                if lineList[0] == "End":
                    joint_name.append(joint_name[stack[-1]] + "_end")
                else:
                    joint_name.append(lineList[1])

                joint_parent.append(stack[-1])
                stack.append(len(joint_name) - 1)
                assert (len(joint_parent) == len(joint_name))

            elif lineList[0] == "OFFSET":
                pos = []
                for j in range(1, 1 + dimension):
                    pos.append(float(lineList[j]))
                joint_offset.append(pos)
                assert(len(joint_offset) == len(joint_name))

            elif lineList[0] == "}":
                stack.pop()

            i += 1

    joint_offset = np.array(joint_offset)
    return joint_name, joint_parent, joint_offset


def part2_forward_kinematics(joint_name, joint_parent, joint_offset, motion_data, frame_id):
    """请填写以下内容
    输入: part1 获得的关节名字，父节点列表，偏移量列表
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数
        frame_id: int，需要返回的帧的索引
    输出:
        joint_positions: np.ndarray，形状为(M, 3)的numpy数组，包含着所有关节的全局位置
        joint_orientations: np.ndarray，形状为(M, 4)的numpy数组，包含着所有关节的全局旋转(四元数)
    Tips:
        1. joint_orientations的四元数顺序为(x, y, z, w)
        2. from_euler时注意使用大写的XYZ
    """

    frame = motion_data[frame_id]
    joint_positions = []
    joint_orientations = []
    frameIdx = 0

    for i in range(0, len(joint_name)):
        name = joint_name[i]

        channels = -1
        position = None
        orientation = None

        if name == "RootJoint":
            channels = 6

            position = joint_offset[i] + [frame[frameIdx], frame[frameIdx + 1], frame[frameIdx + 2]]
            orientation = R.from_euler('XYZ', [frame[frameIdx + 3], frame[frameIdx + 4], frame[frameIdx + 5]],
                                       degrees=True)
            orientation = orientation.as_quat()

        elif name.find("_end") != -1:
            channels = 0
            parentIdx = joint_parent[i]
            orientation = joint_orientations[parentIdx]
            position = joint_positions[parentIdx] + R.from_quat(joint_orientations[parentIdx]).apply(joint_offset[i])

        else:
            channels = 3
            parentIdx = joint_parent[i]
            position = joint_positions[parentIdx] + R.from_quat(joint_orientations[parentIdx]).apply(joint_offset[i])
            rotation = R.from_euler('XYZ', [frame[frameIdx], frame[frameIdx + 1], frame[frameIdx + 2]], degrees=True)

            parentOrientation = R.from_quat(joint_orientations[parentIdx])

            orientation = parentOrientation * rotation
            orientation = orientation.as_quat()

        joint_positions.append(position)
        joint_orientations.append(orientation)
        frameIdx += channels



    assert(len(joint_name) == len(joint_positions))
    assert(len(joint_name) == len(joint_orientations))
    joint_positions = np.array(joint_positions)
    joint_orientations = np.array(joint_orientations)
    return joint_positions, joint_orientations


def find_rotation(vec):
    """
    input: a vector
    output: a quaternion that rotate (1, 0, 0) to this vector
    """
    z_rot = 0
    y_rot = 0

    if vec[0] == 0 and vec[1] == 0 and vec[2] == 0:
        return R.identity().as_quat()

    if vec[0] == 0 and vec[1] == 0:
        if vec[2] > 0:
            return R.from_euler("Y", [-90.0], degrees=True)
        else:
            return R.from_euler("Y", [90.0], degrees=True)

    if vec[0] == 0:
        z_rot = 0
    else:
        z_rot = math.atan(vec[1]/vec[0])

    y_rot = -1 * math.atan(vec[2]/math.sqrt(vec[0] * vec[0] + vec[1] * vec[1]))

    return R.from_euler("ZY", [z_rot, y_rot]).as_quat()


def part3_retarget_func(T_pose_bvh_path, A_pose_bvh_path):
    """
    将 A-pose的bvh重定向到T-pose上
    输入: 两个bvh文件的路径
    输出: 
        motion_data: np.ndarray，形状为(N,X)的numpy数组，其中N为帧数，X为Channel数。retarget后的运动数据
    Tips:
        两个bvh的joint name顺序可能不一致哦(
        as_euler时也需要大写的XYZ
    """
    T_joint_name, T_joint_parent, T_joint_offset = part1_calculate_T_pose(T_pose_bvh_path)
    A_joint_name, A_joint_parent, A_joint_offset = part1_calculate_T_pose(A_pose_bvh_path)



    assert(len(T_joint_name) == len(A_joint_name))
    A_to_T = [-1] * len(A_joint_name)
    T_to_A = [-1] * len(T_joint_name)

    # find the mapping between joints on A pose and T pose
    for i in range(0, len(A_joint_name)):
        for j in range(0, len(T_joint_name)):
            if A_joint_name[i] == T_joint_name[j]:
                A_to_T[i] = j
                T_to_A[j] = i

    # find the index position of each joint in the frame
    start_index = []

    cur_idx = 0
    for i in range(0, len(A_joint_name)):
        name = A_joint_name[i]

        if name == "RootJoint":
            start_index.append(0)
            cur_idx += 6
        elif name.find("_end") != -1:
            start_index.append(-1)
        else:
            start_index.append(cur_idx)
            cur_idx += 3


    # find global orientation offset from A to T
    # we calculate the orientation of a joint based on its offset, but this is actually
    # its parent orientation, thus every time we want to access the transformation
    # we will add + 1 to the index
    A_to_T_offset = []
    for i in range(0, len(A_joint_name)):
        T_index = A_to_T[i]
        A_orientation = R.from_quat(find_rotation(A_joint_offset[i]))
        T_orientation = R.from_quat(find_rotation(T_joint_offset[T_index]))
        A_to_T_offset.append((T_orientation * A_orientation.inv()).as_quat())

    motion_data = load_motion_data(A_pose_bvh_path)

    retargeted_motion_data = []

    for frame_idx in range(0, len(motion_data)):
        print(frame_idx)
        frame = motion_data[frame_idx]
        new_frame = []

        # adjust root joint position and orientation
        A_pos = np.array([frame[0], frame[1], frame[2]])
        T_pos = np.array(A_joint_offset[0]) + A_pos - np.array(T_joint_offset[0])
        new_frame += T_pos.tolist()

        A_rot = R.from_euler("XYZ", [frame[3], frame[4], frame[5]], degrees=True)
        T_rot = A_rot * R.from_quat(A_to_T_offset[0]).inv()
        new_frame += T_rot.as_euler('XYZ', degrees=True).tolist()


        for index_t in range(1, len(T_joint_name)):
            index_a = T_to_A[index_t]


            name = T_joint_name[index_t]

            if name.find("_end") != -1:
                continue
            #new_frame += [frame[start_index[index_a]], frame[start_index[index_a] + 1], frame[start_index[index_a] + 2]]
            #continue
            # when it is a normal joint, that has XYZ rotation channels
            start = start_index[index_a]
            rotation_a = R.from_euler('XYZ', [frame[start], frame[start + 1], frame[start + 2]], degrees=True)

            # For reason why need +1, see comment 228 when constructing A_to_T_offset
            orientation_a_t_i = R.from_quat(A_to_T_offset[index_a + 1])
            orientation_a_t_pi = R.from_quat(A_to_T_offset[A_joint_parent[index_a] + 1])
            rotation_t = orientation_a_t_pi * rotation_a * orientation_a_t_i.inv()

            new_frame += rotation_t.as_euler("XYZ", degrees=True).tolist()

        retargeted_motion_data.append(new_frame)

    return np.array(retargeted_motion_data)
