import numpy as np
from irsim.env import EnvBase
import time
from SimpleMapper import FixedMapper 
from TebSolver import TebplanSolver

import cv2
from sklearn.cluster import DBSCAN


class SIM_ENV:
    def __init__(self, world_file="robot_world.yaml", render=False):
        # 初始化环境
        self.env = EnvBase(world_file, display=render, disable_all_plot=not render,save_ani = True)

        # 地图参数
        map_param = self.env.get_map()
        self.map_width = map_param.width
        self.map_height = map_param.height
        self.map_resolution = map_param.resolution
        print(f"地图尺寸: {self.map_width}m x {self.map_height}m, 分辨率: {self.map_resolution}m")
        
        # 激光雷达最大测距的 80%
        self.lidar_r = self.env.get_lidar_scan()['range_max'] * 0.8  
        
        # 局部求解器
        self.solver = TebplanSolver(np.array([0.0,0.0,0.0]),np.array([0.0,0.0,0.0]),obstacles=None)

        # 固定尺寸建图器
        self.mapper = FixedMapper(width_m=self.map_width, height_m=self.map_height, resolution=self.map_resolution*2)
        
        # 速度指令
        self.action_input = np.array([[0.0], [0.0]]) 
        
    def step(self):
        # 环境单步仿真
        self.env.step(action_id=0, action=self.action_input)

        robot_state = self.env.get_robot_state() 
        scan_data = self.env.get_lidar_scan()

        # 获取机器人姿态及环境信息
        robot_state = self.env.get_robot_state()
        scan_data = self.env.get_lidar_scan()
        obs_list, center_list = self.scan_ellipse(robot_state,scan_data)
        
        # 绘制障碍
        for obs in obs_list:
            self.env.draw_box(obs, refresh=True, color= "-b")

        # 4. 更新地图
        if robot_state is not None and scan_data is not None:
            # FixedMapper 的 world_to_grid 会根据 rx, ry 直接对应到栅格
            self.mapper.update_map(robot_state, scan_data)
            # 获取探索统计数据
            exp_rate, remain_unknown = self.mapper.get_exploration_stats()     
            
            # 停止条件：如果没有未知栅格（或者达到极高比例，如 99.9%）
            if remain_unknown == 0:
                print("全地图探索完成，程序自动停止。")
                # 停止前绘制最后一次完整地图
                self.env.env_config._env_plot.draw_grid_map(self.mapper.get_render_map())
                self.env.render()
                return True

            if self.env.time % 0.5 == 0:
                print(f"运行时间: {self.env.time} |探索率: {exp_rate:.2%} | 剩余未知栅格: {remain_unknown}")
                prob_map = self.mapper.get_render_map()
                self.env.env_config._env_plot.draw_grid_map(prob_map)

            frontier_points = self.mapper.get_closest_frontiers(robot_state)
            last_frontier_points = frontier_points
            if frontier_points is not None:
                target_xy = frontier_points[:2, 0] # 取最近的一个点
                # env.draw_points(obs_list.T, s=10, c='g', refresh=True)obs_list
                # 绘制边界点（红色）
                self.env.draw_points(target_xy.reshape(2,1), s=40, c='r', refresh=True)
            else:
                frontier_points = last_frontier_points

            # 求解局部最优轨迹
            first_target = frontier_points[:, 0]
            traj, dt_seg = self.solver.solve(robot_state.squeeze(), first_target, center_list)   
            traj_xy = traj[:, :2]    

            # 轨迹可视化
            traj_list = [np.array([[xy[0]], [xy[1]]]) for xy in traj_xy]
            self.env.draw_trajectory(traj_list, 'g--', refresh=True)

            # 计算速度指令作为下次仿真输入
            self.compute_v_omega(traj[0,:] ,traj[1,:], dt_seg[0])
            


        # 环境可视化
        if self.env.display:
            self.env.render()


        
        # 是否碰撞
        if self.env.robot.collision:
            print("collision !!!")
            return True
        
        return False
    
    def compute_v_omega(self, p0, p1, dt):
        x0, y0, th0 = p0
        x1, y1, th1 = p1
        # 1) 线速度（带方向）
        dx = x1 - x0
        dy = y1 - y0
        v = (dx * np.cos(th0) + dy * np.sin(th0)) / dt  # 沿当前朝向的投影速度
        # 2) 角速度（带方向）
        dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))  # 最短方向 [-π, π]
        w = dth / dt
        self.action_input = np.array([[float(v)], [float(w)]])

    def compute_currentGoal(self, robot_state):
        rx, ry = robot_state[0], robot_state[1]
        path = self.global_path 
        goal_index = 0

        # 1. 计算所有点到机器人的距离
        robot_xy = robot_state.reshape(-1)  # 将(2,1)重塑为(2,)
        
        # 计算路径上每个点到机器人位置的距离
        dists = np.linalg.norm(path - robot_xy[:2], axis=1)

        # 2. 更新 path_index 为最近点索引（防止倒退）
        nearest_idx = int(np.argmin(dists))
        self.path_index = max(self.path_index, nearest_idx)

        # 3. 从 path_index 开始找第一个距离 > lidar_r 的点
        found = False
        for i in range(self.path_index, len(path)):
            if dists[i] > self.lidar_r:
                goal_index = i
                found = True
                break
        
        # 4. 确定最终目标点
        if not found:
            # 如果没找到，使用全局目标点
            goal_index = len(path) - 1
            target_x, target_y = path[goal_index]
            target_theta = self.robot_goal[-1]
        else:
            # 如果找到，使用路径上的点
            target_x, target_y = path[goal_index]
            target_theta = np.arctan2(target_y - ry, target_x - rx)

        # 返回目标点和朝向
        return np.array([[target_x], [target_y], target_theta])

    
    def scan_ellipse(self, state, scan_data):
            ranges = np.array(scan_data['ranges'])
            angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

            point_list = []
            obstacle_list = []
            center_list = []  # 在这里，center_list 将存储椭圆的完整参数 [cx, cy, a, b, theta]

            for i in range(len(ranges)):
                scan_range = ranges[i]
                angle = angles[i]

                if scan_range < (scan_data['range_max'] - 0.1):
                    # 激光雷达坐标系下的点
                    point = np.array([[scan_range * np.cos(angle)], [scan_range * np.sin(angle)]])
                    point_list.append(point)

            if len(point_list) < 5:  # 拟合椭圆至少需要5个点
                return obstacle_list, center_list

            else:
                point_array = np.hstack(point_list).T
                # 使用 DBSCAN 聚类
                labels = DBSCAN(eps=0.2, min_samples=3).fit_predict(point_array)

                for label in np.unique(labels):
                    if label == -1:
                        continue
                    
                    point_array2 = point_array[labels == label]
                    
                    # 拟合椭圆需要至少 5 个非共线点
                    if len(point_array2) < 5:
                        continue

                    # 1. 拟合局部坐标系下的椭圆
                    # ellipse 返回格式: ((中心x, 中心y), (长短轴直径w, h), 旋转角度deg)
                    ellipse = cv2.fitEllipse(point_array2.astype(np.float32))
                    (lc_x, lc_y), (w, h), angle_deg = ellipse

                    # 2. 坐标变换：从机器人局部坐标系转到全局坐标系
                    trans = state[0:2] # [x, y]
                    rot = state[2, 0]  # theta
                    R = np.array([[np.cos(rot), -np.sin(rot)], 
                                [np.sin(rot),  np.cos(rot)]])

                    # 转换中心点
                    center_local = np.array([[lc_x], [lc_y]])
                    center_global = trans + R @ center_local

                    # 转换旋转角 (OpenCV 角度是顺时针，需注意与机器人坐标系的映射)
                    # 全局角度 = 局部椭圆角度 + 机器人当前朝向
                    angle_rad = np.deg2rad(angle_deg) + rot

                    obs_a = max(w / 2.0, 0.05)
                    obs_b = max(h / 2.0, 0.05)
                    
                    # --- 预计算三角函数 ---
                    cos_ot = np.cos(angle_rad)
                    sin_ot = np.sin(angle_rad)
                    
                    # 扩展返回参数：[cx, cy, a, b, theta, cos_theta, sin_theta]
                    ellipse_params = [
                        center_global[0, 0], 
                        center_global[1, 0], 
                        obs_a, 
                        obs_b, 
                        angle_rad,
                        cos_ot,
                        sin_ot
                    ]
                    center_list.append(ellipse_params)

                    # 4. 为了可视化，仍然可以计算矩形顶点（用于 draw_box）
                    rect = cv2.minAreaRect(point_array2.astype(np.float32))
                    box = cv2.boxPoints(rect)
                    global_vertices = trans + R @ box.T
                    obstacle_list.append(global_vertices)

                return obstacle_list, center_list