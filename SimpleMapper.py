import numpy as np
import irsim
from numba import njit
from sklearn.cluster import DBSCAN

# --- JIT 加速算法 ---

@njit(cache=True)
def njit_log_odds_update(log_odds_grid, map_width, map_height, grid_x, grid_y, 
                         is_occupied, log_odds_occ, log_odds_free, 
                         l_free_threshold, L_MIN, L_MAX):
    if 0 <= grid_x < map_width and 0 <= grid_y < map_height:
        l_current = log_odds_grid[grid_y, grid_x]
        if is_occupied:
            l_update = log_odds_occ
        else:
            if l_current >= l_free_threshold: return 
            l_update = log_odds_free
        l_new = l_current + l_update
        log_odds_grid[grid_y, grid_x] = max(L_MIN, min(l_new, L_MAX))

@njit(cache=True)
def njit_bresenham_line(x0, y0, x1, y1):
    points = []
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x0 < x1 else -1), (1 if y0 < y1 else -1)
    err = dx - dy
    while True:
        points.append((x0, y0))
        if x0 == x1 and y0 == y1: break
        e2 = 2 * err
        if e2 > -dy: err -= dy; x0 += sx
        if e2 < dx: err += dx; y0 += sy
    return points

class FixedMapper:
    def __init__(self, width_m=10.0, height_m=10.0, resolution=0.1):
        self.res = resolution
        self.width_cells = int(width_m / resolution)   
        self.height_cells = int(height_m / resolution) 
        
        # 初始 Log-Odds 地图 (0.0 = 0.5 概率)
        self.log_odds_grid = np.zeros((self.height_cells, self.width_cells), dtype=np.float32)
        
        # 概率参数
        self.l_occ = np.log(0.75 / 0.25)   
        self.l_free = np.log(0.3 / 0.7)  
        self.l_max, self.l_min = 4.59, -4.59 
        self.l_threshold = 0.6

    def world_to_grid(self, x, y):
        # 考虑到原点在左下角，直接映射。添加 clip 防止超出 10x10 范围
        gx = int(np.clip(x / self.res, 0, self.width_cells - 1))
        gy = int(np.clip(y / self.res, 0, self.height_cells - 1))
        return gx, gy

    def update_map(self, robot_pos, scan_data):
        # robot_pos: [x, y, theta]
        rx, ry, r_theta = robot_pos[0,0], robot_pos[1,0], robot_pos[2,0]
        rgx, rgy = self.world_to_grid(rx, ry)

        ranges = np.array(scan_data['ranges'])
        angles = np.linspace(scan_data['angle_min'], scan_data['angle_max'], len(ranges))
        
        for r, angle in zip(ranges, angles):
            if r < scan_data['range_min'] or np.isnan(r): continue
            
            is_obs = r < (scan_data['range_max'] - 0.1)
            world_angle = r_theta + angle
            tx = rx + r * np.cos(world_angle)
            ty = ry + r * np.sin(world_angle)
            tgx, tgy = self.world_to_grid(tx, ty)

            path = njit_bresenham_line(rgx, rgy, tgx, tgy)
            
            # 更新空闲路径
            for i in range(len(path) - 1):
                njit_log_odds_update(self.log_odds_grid, self.width_cells, self.height_cells, 
                                     path[i][0], path[i][1], False, 
                                     self.l_occ, self.l_free, self.l_threshold, self.l_min, self.l_max)
            
            # 更新障碍物终点
            if is_obs:
                njit_log_odds_update(self.log_odds_grid, self.width_cells, self.height_cells, 
                                     tgx, tgy, True, 
                                     self.l_occ, self.l_free, self.l_threshold, self.l_min, self.l_max)

    def get_render_map(self):
        # 转换回 0-1 概率供绘制
        return 1.0 / (1.0 + np.exp(-self.log_odds_grid.T))
    
    def get_exploration_stats(self):
            """
            计算探索进度
            返回: (已探索比例, 剩余未知栅格数)
            """
            # 未知栅格定义为：概率在 0.45 到 0.55 之间（即 log-odds 接近 0）
            # 或者更严格一点：log_odds_grid == 0
            unknown_mask = (self.log_odds_grid == 0)
            unknown_count = np.sum(unknown_mask)
            total_count = self.width_cells * self.height_cells
            
            explored_count = total_count - unknown_count
            exploration_rate = explored_count / total_count
            
            return exploration_rate, unknown_count
    
    def get_frontiers(self):
            """
            优化后的边界点检测：增加检测灵敏度并扩大搜索范围
            """
            import numpy as np
            from scipy.ndimage import binary_dilation

            # 1. 放宽“未知区域”的定义
            # 原本只有 == 0 才算未知，现在将靠近 0 的微小变化区域也视为未知
            # 这样可以捕获更多处于探索边缘的栅格
            unknown_mask = (self.log_odds_grid >= -0.2) & (self.log_odds_grid <= 0.2)
            
            # 已知空闲区域（保持原样或稍微放宽）
            free_mask = self.log_odds_grid < -0.4

            # 2. 增强邻域搜索 (关键修改)
            # 使用更大的结构元素进行膨胀，或者增加迭代次数
            # iterations=2 会让寻找范围向外扩张，从而检测到更多点
            dilated_unknown = binary_dilation(unknown_mask, iterations=1)

            # 既是 Free，又挨着（或靠近）Unknown
            frontier_mask = free_mask & dilated_unknown

            # 获取坐标索引 (y, x)
            y_indices, x_indices = np.where(frontier_mask)

            if len(x_indices) == 0:
                return None

            # 3. 转换为世界坐标
            # 建议检查是否需要加上地图的原点偏移 self.origin
            # 这里假设 self.origin 存在，若不存在请删掉 offset 部分
            offset_x = self.origin[0] if hasattr(self, 'origin') else 0
            offset_y = self.origin[1] if hasattr(self, 'origin') else 0

            x_m = x_indices * self.res + offset_x
            y_m = y_indices * self.res + offset_y

            # 4. 包装并返回
            return np.vstack((x_m, y_m))

    def get_frontiers_smart(self, eps_m=1.0, min_samples=2):
        """
        使用聚类提取边界区域的中心点
        eps_m: 聚类半径（米），例如 0.3 米内的点聚为一类
        min_samples: 每个簇最少需要的点数（过滤孤立噪声点）
        """
        # 1. 按照你之前的逻辑获取所有原始点
        raw_frontiers = self.get_frontiers() # 即你刚才写的函数逻辑
        if raw_frontiers is None: return None
        
        # 转换为 (N, 2) 形状以供 sklearn 使用
        points = raw_frontiers.T 
        
        # 2. 聚类计算 (eps 转化为像素单位或直接在物理坐标上做)
        clustering = DBSCAN(eps=eps_m, min_samples=min_samples).fit(points)
        labels = clustering.labels_
        
        centroids = []
        unique_labels = set(labels)
        for label in unique_labels:
            if label == -1: continue  # 跳过噪声
            
            # 提取属于该簇的所有点并计算平均值（质心）
            class_member_mask = (labels == label)
            centroid = points[class_member_mask].mean(axis=0)
            centroids.append(centroid)
            
        if not centroids: return None
        
        # 3. 返回 (2, K) 形状，K 是簇的数量
        return np.array(centroids).T

    def get_closest_frontiers(self, robot_pos, num_targets=5):
            all_frontiers = self.get_frontiers() 
            
            if all_frontiers is None or all_frontiers.size == 0: 
                self.current_frontier_target = None # 清除记忆
                return None
                
            rx, ry, r_theta = float(robot_pos[0,0]), float(robot_pos[1,0]), float(robot_pos[2,0])
            
            dx = all_frontiers[0, :] - rx
            dy = all_frontiers[1, :] - ry
            
            dists = np.sqrt(dx**2 + dy**2)
            target_thetas = np.arctan2(dy, dx)
            angle_diffs = np.abs(np.arctan2(np.sin(target_thetas - r_theta), np.cos(target_thetas - r_theta)))
            
            candidate_scores = []
            safe_distance_m = 0.5 
            
            # --- 核心修改：目标持久化权重 ---
            # 如果当前已经有目标，给靠近旧目标的点一个“奖励”（负代价）
            # 防止因微小误差导致的频繁切换
            last_target = getattr(self, 'current_frontier_target', None)

            for i in range(all_frontiers.shape[1]):
                tx, ty = all_frontiers[0, i], all_frontiers[1, i]
                
                # 1. 基础惩罚项
                obs = self.get_obstacles_on_line([rx, ry], [tx, ty])
                obs_count = obs.shape[0] if obs is not None else 0
                
                ix, iy = int(tx / self.res), int(ty / self.res)
                R = int(safe_distance_m / self.res)
                y_min, y_max = max(0, iy-R), min(self.log_odds_grid.shape[0], iy+R+1)
                x_min, x_max = max(0, ix-R), min(self.log_odds_grid.shape[1], ix+R+1)
                local_map = self.log_odds_grid[y_min:y_max, x_min:x_max]
                
                proximity_penalty = 50.0 if np.any(local_map > 0.4) else 0
                
                # 2. 计算持久化奖励 (Hysteresis)
                hysteresis_reward = 0
                if last_target is not None:
                    # 计算当前候选点与上次选定目标的距离
                    dist_to_last = np.sqrt((tx - last_target[0])**2 + (ty - last_target[1])**2)
                    if dist_to_last < 1.0: # 如果该点就在旧目标附近 1米内
                        hysteresis_reward = -2.0 # 给予大幅度减分奖励，使其优先被选中

                # 3. 综合评分
                total_cost = dists[i] + angle_diffs[i] + (obs_count * 10.0) + proximity_penalty + hysteresis_reward
                candidate_scores.append((total_cost, tx, ty, target_thetas[i]))

            # 排序
            candidate_scores.sort(key=lambda x: x[0])
            
            # 记录本次的最优目标，供下次调用参考
            if len(candidate_scores) > 0:
                best = candidate_scores[0]
                self.current_frontier_target = [best[1], best[2]] # 存储 [tx, ty]

            best_targets = []
            for i in range(min(num_targets, len(candidate_scores))):
                _, tx, ty, theta = candidate_scores[i]
                best_targets.append([tx, ty, theta])
                
            return np.array(best_targets).T
            
    def get_obstacles_on_line(self, start_world, end_world):
            """
            获取从机器人到边界点连线上的障碍点。
            
            返回: 
                - np.ndarray (N, 2): 障碍物坐标
                - None: 如果路径上没有任何障碍
            """
            # 1. 将世界坐标转换为栅格索引 (注意处理 robot_state 可能是 (3,1) 的情况)
            x0, y0 = self.world_to_grid(float(start_world[0]), float(start_world[1]))
            x1, y1 = self.world_to_grid(float(end_world[0]), float(end_world[1]))

            # 2. 利用 Bresenham 算法获取线段上的所有栅格坐标
            path_indices = njit_bresenham_line(x0, y0, x1, y1)

            obstacle_points = []
            
            # 3. 检查路径上的每个栅格
            for gx, gy in path_indices:
                # 这里的阈值 self.l_threshold (0.85) 用于判定是否为障碍
                if self.log_odds_grid[gy, gx] >= self.l_threshold:
                    # 转换回物理世界坐标 (x, y)
                    ox = gx * self.res
                    oy = gy * self.res
                    obstacle_points.append([ox, oy])
            
            # 4. 关键：格式转换与返回
            if len(obstacle_points) > 0:
                # 转换为 numpy 数组，形状自动为 (N, 2)
                # 建议对点云进行下采样，TEB Solver 对太多障碍点会变慢
                return np.array(obstacle_points)
            else:
                return None
# --- 主程序逻辑 ---

def main():
    # 1. 启动仿真
    env = irsim.make(save_ani=False, full=False)
    
    # 2. 【关键】获取初始位姿以确认环境状况（即便不移动）
    initial_state = env.get_robot_state()
    if initial_state is not None:
        print(f"机器人初始位置: x={initial_state[0,0]:.2f}, y={initial_state[1,0]:.2f}")

    # 3. 初始化 10m x 10m 的固定建图器
    mapper = FixedMapper(width_m=10.0, height_m=10.0, resolution=0.1)
    
    # 简单控制指令
    vel = np.array([[0.5], [0.1]]) 

    print("开始建图逻辑...")

    for i in range(2000):
        env.step(vel)
        
        robot_state = env.get_robot_state()
        scan_data = env.get_lidar_scan()

        if robot_state is not None and scan_data is not None:
            # 即使初始位置不在(0,0)，world_to_grid 也会正确处理
            mapper.update_map(robot_state, scan_data)
            
            # 渲染地图
            prob_map = mapper.get_render_map()
            
            # ir-sim 绘制
            # 如果发现画面上下反了，请改用 env.env_config._env_plot.draw_grid_map(np.flipud(prob_map))
            env.env_config._env_plot.draw_grid_map(prob_map)

        env.render()
        if env.done(): break

    env.end()

if __name__ == '__main__':
    main()