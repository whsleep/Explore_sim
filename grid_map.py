import irsim
import numpy as np
import time
from SimpleMapper import FixedMapper 
from TebSolver import TebplanSolver


def compute_v_omega(p0, p1, dt):
    x0, y0, th0 = p0
    x1, y1, th1 = p1
    # 1) 线速度（带方向）
    dx = x1 - x0
    dy = y1 - y0
    v = (dx * np.cos(th0) + dy * np.sin(th0)) / dt  # 沿当前朝向的投影速度
    # 2) 角速度（带方向）
    dth = np.arctan2(np.sin(th1 - th0), np.cos(th1 - th0))  # 最短方向 [-π, π]
    w = dth / dt
    return v,w

def scan_process(state, scan_data):
        """
        处理扫描数据，返回符合 TebSolver 要求的 (N, 2) 形状数组
        """
        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))

        # 提取机器人当前位姿
        rx, ry, r_theta = state[0, 0], state[1, 0], state[2, 0]

        point_list = []

        for i in range(len(ranges)):
            scan_range = ranges[i]
            angle = angles[i]

            # 过滤无效点
            if scan_range < (scan_data['range_max'] - 0.1) and scan_range > 0.15:
                # 局部坐标转世界坐标
                local_x = scan_range * np.cos(angle)
                local_y = scan_range * np.sin(angle)
                
                world_x = rx + local_x * np.cos(r_theta) - local_y * np.sin(r_theta)
                world_y = ry + local_x * np.sin(r_theta) + local_y * np.cos(r_theta)
                
                point_list.append([world_x, world_y])

        # 1. 如果没有检测到障碍物，返回一个 (0, 2) 的空数组
        # 这样 solver 里的 for (ox, oy) in obs_now 循环会直接跳过而不会报错
        if not point_list:
            return None
        # 2. 将点转换为 (N, 2) 形状的 numpy 数组
        point_array = np.array(point_list)

        # 3. 建议进行简单的下采样，防止激光点过多导致求解器变慢
        # 每隔 3 个点取一个
        return point_array

def run_mapping():
    # 1. 初始化仿真环境
    # 假设你的环境 yaml 定义的是 10x10 的空间
    env = irsim.make(save_ani=True, full=False)
    
    # 2. 获取机器人的初始状态 (用于确认位置，即便不手动传入初始化)
    init_robot_state = env.get_robot_state()
    print(f"机器人初始位置: {init_robot_state[:2, 0]}")

    # 3. 初始化 FixedMapper
    # 设置为固定 10m x 10m，分辨率 0.2 (生成 50x50 矩阵)
    mapper = FixedMapper(width_m=10.0, height_m=10.0, resolution=0.2)

    # 
    solver = TebplanSolver(np.array([1.0,1.0,0.0]),np.array([0.0,0.0,0.0]),obstacles = None)
    v,w = 0.0,0.0

    for i in range(20000):
        # 机器人运动控制
        env.step(action=np.array([[float(v)], [float(w)]])) 
        v,w = 0.0,0.0        
        # 获取最新的仿真数据
        robot_state = env.get_robot_state() # 得到的是实时的 [x, y, theta]
        scan_data = env.get_lidar_scan()
        obs_list = scan_process(robot_state, scan_data)
        
        # 4. 更新地图
        if robot_state is not None and scan_data is not None:
            # FixedMapper 的 world_to_grid 会根据 rx, ry 直接对应到栅格
            mapper.update_map(robot_state, scan_data)
            # 获取探索统计数据
            exp_rate, remain_unknown = mapper.get_exploration_stats()     
            
            # 停止条件：如果没有未知栅格（或者达到极高比例，如 99.9%）
            if remain_unknown == 0:
                    print("全地图探索完成，程序自动停止。")
                    # 停止前绘制最后一次完整地图
                    env.env_config._env_plot.draw_grid_map(mapper.get_render_map())
                    env.render()
                    time.sleep(2)
                    break

        # 5. 获取概率地图并绘制
        # 降低绘制频率以解决耗时问题（每 100 步重绘一次可视化）
        if i % 10 == 0:
            print(f"步数: {i} | 探索率: {exp_rate:.2%} | 剩余未知栅格: {remain_unknown}")
            prob_map = mapper.get_render_map()
            # 使用 .T 或 np.flipud 确保坐标轴方向正确
            env.env_config._env_plot.draw_grid_map(prob_map)

        # --- 新增：检测并绘制边界点 ---
        frontier_points = mapper.get_closest_frontiers(robot_state)
        last_frontier_points = frontier_points
        if frontier_points is not None:
            target_xy = frontier_points[:2, 0] # 取最近的一个点
            # env.draw_points(obs_list.T, s=10, c='g', refresh=True)obs_list
            # 绘制边界点（红色）
            env.draw_points(target_xy.reshape(2,1), s=40, c='r', refresh=True)
        else:
            frontier_points = last_frontier_points

        # 求解局部最优轨迹
        first_target = frontier_points[:, 0]
        traj, dt_seg = solver.solve(robot_state.squeeze(), first_target, obs_list)   
        traj_xy = traj[:, :2]         

        # 轨迹可视化
        traj_list = [np.array([[xy[0]], [xy[1]]]) for xy in traj_xy]
        env.draw_trajectory(traj_list, 'g--', refresh=True)

        # 计算速度指令作为下次仿真输入
        v,w = compute_v_omega(traj[0,:] ,traj[1,:], dt_seg[0])

        env.render()
        
        if env.done():
            break

    env.end()

if __name__ == '__main__':
    run_mapping()