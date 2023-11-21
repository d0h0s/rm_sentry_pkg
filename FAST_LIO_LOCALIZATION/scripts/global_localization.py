#!/usr/bin/env python3
# coding=utf8
from __future__ import print_function, division, absolute_import

import _thread
import copy
import time

import numpy as np
import open3d as o3d
import ros_numpy
import rospy
import tf
import tf.transformations
from geometry_msgs.msg import PoseWithCovarianceStamped, Pose, Point, Quaternion
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2

global_map = None
initialized = False
T_map_to_odom = np.eye(4)
cur_odom = None
cur_scan = None
relocate_times = 0

def pose_to_mat(pose_msg):
    return np.matmul(
        tf.listener.xyz_to_mat44(pose_msg.pose.pose.position),
        tf.listener.xyzw_to_mat44(pose_msg.pose.pose.orientation),
    )


def msg_to_array(pc_msg):
    pc_array = ros_numpy.numpify(pc_msg)
    pc = np.zeros([len(pc_array), 3])
    pc[:, 0] = pc_array['x']
    pc[:, 1] = pc_array['y']
    pc[:, 2] = pc_array['z']
    return pc


def registration_at_scale(pc_scan, pc_map, initial, scale):
    result_icp = o3d.pipelines.registration.registration_icp(
        voxel_down_sample(pc_scan, SCAN_VOXEL_SIZE * scale), voxel_down_sample(pc_map, MAP_VOXEL_SIZE * scale),
        1.0 * scale, initial,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=20)
    )

    return result_icp.transformation, result_icp.fitness


def inverse_se3(trans):
    trans_inverse = np.eye(4)
    # R
    trans_inverse[:3, :3] = trans[:3, :3].T
    # t
    trans_inverse[:3, 3] = -np.matmul(trans[:3, :3].T, trans[:3, 3])
    return trans_inverse


def publish_point_cloud(publisher, header, pc):
    data = np.zeros(len(pc), dtype=[
        ('x', np.float32),
        ('y', np.float32),
        ('z', np.float32),
        ('intensity', np.float32),
    ])
    data['x'] = pc[:, 0]
    data['y'] = pc[:, 1]
    data['z'] = pc[:, 2]
    if pc.shape[1] == 4:
        data['intensity'] = pc[:, 3]
    msg = ros_numpy.msgify(PointCloud2, data)
    msg.header = header
    publisher.publish(msg)


def crop_global_map_in_FOV(global_map, pose_estimation, cur_odom):
    # 当前scan原点的位姿
    T_odom_to_base_link = pose_to_mat(cur_odom)
    T_map_to_base_link = np.matmul(pose_estimation, T_odom_to_base_link)
    T_base_link_to_map = inverse_se3(T_map_to_base_link)

    # 把地图转换到lidar系下
    global_map_in_map = np.array(global_map.points)
    global_map_in_map = np.column_stack([global_map_in_map, np.ones(len(global_map_in_map))])
    global_map_in_base_link = np.matmul(T_base_link_to_map, global_map_in_map.T).T

    # 将视角内的地图点提取出来
    if FOV > 3.14:
        # 环状lidar 仅过滤距离
        indices = np.where(
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    else:
        # 非环状lidar 保前视范围
        # FOV_FAR>x>0 且角度小于FOV
        indices = np.where(
            (global_map_in_base_link[:, 0] > 0) &
            (global_map_in_base_link[:, 0] < FOV_FAR) &
            (np.abs(np.arctan2(global_map_in_base_link[:, 1], global_map_in_base_link[:, 0])) < FOV / 2.0)
        )
    global_map_in_FOV = o3d.geometry.PointCloud()
    global_map_in_FOV.points = o3d.utility.Vector3dVector(np.squeeze(global_map_in_map[indices, :3]))

    # 发布fov内点云
    header = cur_odom.header
    header.frame_id = 'map'
    publish_point_cloud(pub_submap, header, np.array(global_map_in_FOV.points)[::10])

    return global_map_in_FOV


def global_localization(pose_estimation):
    global global_map, cur_scan, cur_odom, T_map_to_odom, relocate_times
    # 用icp配准
    # print(global_map, cur_scan, T_map_to_odom)
    rospy.loginfo('Global localization by scan-to-map matching......')

    # TODO 这里注意线程安全
    scan_tobe_mapped = copy.copy(cur_scan)

    tic = time.time()

    global_map_in_FOV = crop_global_map_in_FOV(global_map, pose_estimation, cur_odom)
    # 粗配准
    transformation, _ = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=pose_estimation, scale=5)

    # 精配准
    transformation, fitness = registration_at_scale(scan_tobe_mapped, global_map_in_FOV, initial=transformation,
                                                    scale=1)
    toc = time.time()
    rospy.loginfo('Time: {}'.format(toc - tic))
    rospy.loginfo('')

    # 当全局定位成功时才更新map2odom
    if fitness > LOCALIZATION_TH:
        # T_map_to_odom = np.matmul(transformation, pose_estimation)
        T_map_to_odom = transformation

        # 发布map_to_odom
        map_to_odom = Odometry()
        xyz = tf.transformations.translation_from_matrix(T_map_to_odom)
        quat = tf.transformations.quaternion_from_matrix(T_map_to_odom)
        map_to_odom.pose.pose = Pose(Point(*xyz), Quaternion(*quat))
        map_to_odom.header.stamp = cur_odom.header.stamp
        map_to_odom.header.frame_id = 'map'
        pub_map_to_odom.publish(map_to_odom)
        relocate_times = 0
        return True
    else:
        rospy.logwarn('Not match!!!!')
        rospy.logwarn('{}'.format(transformation))
        rospy.logwarn('fitness score:{}'.format(fitness))
        relocate_times += 1
        return False


def voxel_down_sample(pcd, voxel_size):
    try:
        pcd_down = pcd.voxel_down_sample(voxel_size)
    except:
        # for opend3d 0.7 or lower
        pcd_down = o3d.geometry.voxel_down_sample(pcd, voxel_size)
    return pcd_down


def initialize_global_map(pc_msg):
    global global_map

    global_map = o3d.geometry.PointCloud()
    global_map.points = o3d.utility.Vector3dVector(msg_to_array(pc_msg)[:, :3])
    global_map = voxel_down_sample(global_map, MAP_VOXEL_SIZE)
    rospy.loginfo('Global map received.')


def cb_save_cur_odom(odom_msg):
    global cur_odom
    cur_odom = odom_msg


def cb_save_cur_scan(pc_msg):
    global cur_scan
    # 注意这里fastlio直接将scan转到odom系下了 不是lidar局部系
    pc_msg.header.frame_id = 'camera_init'
    pc_msg.header.stamp = rospy.Time().now()
    pub_pc_in_map.publish(pc_msg)

    # 转换为pcd
    # fastlio给的field有问题 处理一下
    pc_msg.fields = [pc_msg.fields[0], pc_msg.fields[1], pc_msg.fields[2],
                     pc_msg.fields[4], pc_msg.fields[5], pc_msg.fields[6],
                     pc_msg.fields[3], pc_msg.fields[7]]
    pc = msg_to_array(pc_msg)

    cur_scan = o3d.geometry.PointCloud()
    cur_scan.points = o3d.utility.Vector3dVector(pc[:, :3])


def thread_localization():
    global T_map_to_odom
    while True:
        # 每隔一段时间进行全局定位
        rospy.sleep(1 / FREQ_LOCALIZATION)
        # TODO 由于这里Fast lio发布的scan是已经转换到odom系下了 所以每次全局定位的初始解就是上一次的map2odom 不需要再拿odom了
        global_localization(T_map_to_odom)


if __name__ == '__main__':
    MAP_VOXEL_SIZE = 0.4
    SCAN_VOXEL_SIZE = 0.1

    # Global localization frequency (HZ)
    FREQ_LOCALIZATION = 0.5

    # The threshold of global localization,
    # only those scan2map-matching with higher fitness than LOCALIZATION_TH will be taken
    LOCALIZATION_TH = 0.95

    # FOV(rad), modify this according to your LiDAR type
    FOV = 360

    # The farthest distance(meters) within FOV
    FOV_FAR = 150

    rospy.init_node('fast_lio_localization')
    rospy.loginfo('Localization Node Inited...')

    # publisher
    pub_pc_in_map = rospy.Publisher('/cur_scan_in_map', PointCloud2, queue_size=1)
    pub_submap = rospy.Publisher('/submap', PointCloud2, queue_size=1)
    pub_map_to_odom = rospy.Publisher('/map_to_odom', Odometry, queue_size=1)

    rospy.Subscriber('/cloud_registered', PointCloud2, cb_save_cur_scan, queue_size=1)
    rospy.Subscriber('/Odometry', Odometry, cb_save_cur_odom, queue_size=1)

    # 初始化全局地图
    rospy.logwarn('Waiting for global map......')
    initialize_global_map(rospy.wait_for_message('/offline_map', PointCloud2))

    # 初始化
    while not initialized:
        # rospy.logwarn('Waiting for initial pose....')
        #
        # # 等待初始位姿
        # pose_msg = rospy.wait_for_message('/initialpose', PoseWithCovarianceStamped)
        # initial_pose = pose_to_mat(pose_msg)
        # if cur_scan:
        #     initialized = global_localization(initial_pose)
        # else:
        #     rospy.logwarn('First scan not received!!!!!')
        pose_x = rospy.get_param("initial_pose/pose_x", 0.0)
        pose_y = rospy.get_param("initial_pose/pose_y", 0.0)
        pose_z = rospy.get_param("initial_pose/pose_z", 0.0)
        pose_roll = rospy.get_param("initial_pose/pose_roll", 0.0)
        pose_pitch = rospy.get_param("initial_pose/pose_pitch", 0.0)
        pose_yaw = rospy.get_param("initial_pose/pose_yaw", 0.0)

        # 转换为pose
        quat = tf.transformations.quaternion_from_euler(pose_roll, pose_pitch, pose_yaw)
        xyz = [pose_x, pose_y, pose_z]

        pose_msg = PoseWithCovarianceStamped()
        pose_msg.pose.pose = Pose(Point(*xyz), Quaternion(*quat))

        initial_pose = pose_to_mat(pose_msg)
        if cur_scan:
            initialized = global_localization(initial_pose)

    rospy.loginfo('')
    rospy.loginfo('Initialize successfully!!!!!!')
    rospy.loginfo('')
    # 开始定期全局定位
    _thread.start_new_thread(thread_localization, ())

    rospy.spin()

# 这段代码是一个ROS（Robot Operating System）节点，用于在移动机器人上实现全局定位功能。它使用了Python编程语言，主要依赖于Open3D库来处理点云数据以进行全局定位。
#
# 以下是代码的主要部分和功能：
#
# 1. 导入所需的库和模块，包括ROS消息、Open3D、NumPy等。
#
# 2. 定义全局变量 `global_map`，用于存储全局地图点云数据；`initialized` 用于标志是否已经完成初始化；`T_map_to_odom` 存储地图到里程计坐标系的变换；`cur_odom` 存储当前里程计数据；`cur_scan` 存储当前激光扫描数据。
#
# 3. 定义一些辅助函数，如 `pose_to_mat` 用于将位姿消息转换为变换矩阵，`msg_to_array` 用于将ROS消息中的点云数据转换为NumPy数组等。
#
# 4. `registration_at_scale` 函数执行点云配准操作，使用ICP算法将当前扫描数据与全局地图配准，并返回变换矩阵和匹配度。
#
# 5. `inverse_se3` 函数用于计算SE(3)变换的逆变换。
#
# 6. `publish_point_cloud` 函数用于发布点云数据。
#
# 7. `crop_global_map_in_FOV` 函数用于将全局地图数据裁剪为当前激光扫描视场内的点云。
#
# 8. `global_localization` 函数是全局定位的主要逻辑，它首先裁剪全局地图，然后进行点云配准，如果匹配度高于阈值（`LOCALIZATION_TH`），则更新地图到里程计的变换，并发布地图到里程计的变换。
#
# 9. `voxel_down_sample` 函数用于对点云数据进行下采样。
#
# 10. `initialize_global_map` 函数用于初始化全局地图。
#
# 11. `cb_save_cur_odom` 和 `cb_save_cur_scan` 是回调函数，用于保存当前里程计和激光扫描数据。
#
# 12. `thread_localization` 函数是一个线程函数，定期执行全局定位操作。
#
# 13. 在主函数中，定义了一些常量和参数，初始化ROS节点，设置消息发布者和订阅者，等待全局地图数据，执行初始化操作，然后启动全局定位线程。
#
# 此代码的主要目标是在移动机器人上实现全局定位，通过将当前激光扫描数据与全局地图进行匹配，从而估计机器人的全局位置。全局地图的数据来自 `/offline_map` 主题，激光扫描数据来自 `/cloud_registered` 主题。如果全局定位成功，将发布地图到里程计的变换。
