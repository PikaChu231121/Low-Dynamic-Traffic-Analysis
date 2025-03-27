from shapely.geometry import LineString, Point, Polygon
import copy

import numpy as np

from .base_sched import BaseScheduler


class TrafficScheduler(BaseScheduler):
    @staticmethod
    def getConfig(env, name):
        return env.traffic_manager.getConfig(name)

    @staticmethod
    def getCurrentTime(env):
        return env.traffic_manager.getCurrentTime()

    @staticmethod
    def getTrafficInterval(env):
        return env.traffic_interval

    @staticmethod
    def getDistanceBetweenNodesById(env, node_id_1, node_id_2):
        return env.getDistanceBetweenNodesById(node_id_1, node_id_2)

    @staticmethod
    def getUAVTrafficInfos(env):
        return env.traffic_manager.getUAVTrafficInfos()

    @staticmethod
    def getRSUTrafficInfos(env):
        return env.traffic_manager.getRSUInfos()

    @staticmethod
    def setUAVMobilityPatterns(env, UAV_mobility_patterns):
        organized_patterns = {}
        for UAV_id, UAV_mobile_pattern in UAV_mobility_patterns.items():
            organized_patterns[UAV_id] = {}
            organized_patterns[UAV_id]['angle'] = UAV_mobile_pattern['angle']
            organized_patterns[UAV_id]['phi'] = UAV_mobile_pattern['phi']
            organized_patterns[UAV_id]['speed'] = UAV_mobile_pattern['speed']
        env.uav_mobility_patterns = organized_patterns

    @staticmethod
    def getVehicleInfosInRange(env, target_position, distance_threshold):
        target_position = [target_position[0], target_position[1], 0]
        vehicle_infos = env.traffic_manager.getVehicleTrafficInfos()
        candidate_vehicle_infos = {}
        vehicle_ids_list = list(vehicle_infos.keys())
        vehicle_positions = [vehicle_infos[vehicle_id]['position'] for vehicle_id in vehicle_ids_list]
        if len(vehicle_positions) == 0:
            return {}
        vehicle_positions = np.asarray(vehicle_positions)
        distances = np.linalg.norm(vehicle_positions - np.asarray(target_position), axis=1)
        selected_vehicle_ids = np.where(distances <= distance_threshold)[0]
        for idx in selected_vehicle_ids:
            vehicle_id = vehicle_ids_list[idx]
            candidate_vehicle_infos[vehicle_id] = vehicle_infos[vehicle_id]
        return candidate_vehicle_infos

    @staticmethod
    def getRandomTargetPositionForUAV(env, UAV_id):
        # 从env的traffic manager获取当前uav位置；随机找一个不在禁飞区的位置作为目标位置，并且保证两个位置之间的直线也不经过禁飞区
        current_position = env.traffic_manager.getNodePositionById(UAV_id)
        target_positions = env.traffic_manager.getAllJunctionPositions()
        nonfly_zones = env.traffic_manager.getNonFlyZones() # [[[x1,y1],[x2,y2],[x3,y3]], ...]
        
        # 打乱目标位置列表，随机选择
        import random
        random.shuffle(target_positions)
        
        target_position = None
        for pos in target_positions:
            if TrafficScheduler.isPositionInNonFlyZone(pos, nonfly_zones):
                continue
                
            # 检查从当前位置到目标位置的路径是否穿过禁飞区
            path_crosses_nonfly_zone = False
            for nonfly_zone in nonfly_zones:
                if TrafficScheduler.isLineCrossNonFlyZone(current_position, pos, nonfly_zone):
                    path_crosses_nonfly_zone = True
                    break
            
            if not path_crosses_nonfly_zone:
                target_position = pos
                break
                
        return target_position

    @staticmethod
    def isPositionInNonFlyZone(position, nonfly_zones):
        """检查位置是否在禁飞区内"""
        point = Point(position[0], position[1])
        for zone in nonfly_zones:
            polygon = Polygon(zone)
            if polygon.contains(point):
                return True
        return False

    @staticmethod
    def isLineCrossNonFlyZone(start_position, end_position, nonfly_zone):
        """检查线段是否与禁飞区相交或穿过禁飞区"""
        # 创建线段
        line = LineString([(start_position[0], start_position[1]), (end_position[0], end_position[1])])
        # 创建禁飞区多边形
        polygon = Polygon(nonfly_zone)
        
        # 判断线段是否与多边形相交或线段端点在多边形内部
        return line.intersects(polygon) or polygon.contains(Point(start_position[0], start_position[1])) or polygon.contains(Point(end_position[0], end_position[1]))

    @staticmethod
    def planPathAvoidingNonFlyZones(env, start_position, end_position):
        """规划避开禁飞区的路径"""
        nonfly_zones = env.traffic_manager.getNonFlyZones()
        
        # 检查直接路径是否可行
        direct_path_ok = True
        for zone in nonfly_zones:
            if TrafficScheduler.isLineCrossNonFlyZone(start_position, end_position, zone):
                direct_path_ok = False
                break
                
        if direct_path_ok:
            return [start_position, end_position]
            
        # 如果直接路径不可行，使用中间点绕过禁飞区
        waypoints = []
        waypoints.append(start_position)
        
        # 获取所有禁飞区的边界点，用作可能的航点
        potential_waypoints = []
        for zone in nonfly_zones:
            for point in zone:
                # 为边界点添加一定的安全缓冲距离
                buffer_distance = 50  # 安全缓冲距离
                angle = np.random.uniform(0, 2 * np.pi)
                buffered_point = (
                    point[0] + buffer_distance * np.cos(angle),
                    point[1] + buffer_distance * np.sin(angle),
                    start_position[2]  # 保持与起点相同的高度
                )
                if not TrafficScheduler.isPositionInNonFlyZone(buffered_point, nonfly_zones):
                    potential_waypoints.append(buffered_point)
        
        # 添加一些随机的航点
        junctions = env.traffic_manager.getAllJunctionPositions()
        for junction in junctions[:10]:  # 只使用前10个路口作为可能的航点
            if not TrafficScheduler.isPositionInNonFlyZone(junction, nonfly_zones):
                potential_waypoints.append((junction[0], junction[1], start_position[2]))
        
        # 尝试找到一条可行的路径
        current_pos = start_position
        max_attempts = 5
        attempts = 0
        
        while attempts < max_attempts:
            # 按照到目标点的距离排序所有可能的航点
            sorted_waypoints = sorted(potential_waypoints, 
                                      key=lambda wp: np.linalg.norm(np.array([wp[0], wp[1]]) - 
                                                                   np.array([end_position[0], end_position[1]])))
            
            for wp in sorted_waypoints:
                path_ok = True
                # 检查从当前位置到航点的路径是否穿过禁飞区
                for zone in nonfly_zones:
                    if TrafficScheduler.isLineCrossNonFlyZone(current_pos, wp, zone):
                        path_ok = False
                        break
                
                if path_ok:
                    waypoints.append(wp)
                    current_pos = wp
                    # 检查从航点到终点的路径是否穿过禁飞区
                    can_reach_end = True
                    for zone in nonfly_zones:
                        if TrafficScheduler.isLineCrossNonFlyZone(current_pos, end_position, zone):
                            can_reach_end = False
                            break
                    
                    if can_reach_end:
                        waypoints.append(end_position)
                        return waypoints
                    break
            
            attempts += 1
            
        # 如果无法找到完全避开禁飞区的路径，返回包含增高的中间点的路径
        # 增加飞行高度以越过禁飞区
        mid_point = (
            (start_position[0] + end_position[0]) / 2,
            (start_position[1] + end_position[1]) / 2,
            max(start_position[2], end_position[2]) + 100  # 增加高度
        )
        return [start_position, mid_point, end_position]

    @staticmethod
    def getDefaultUAVMobilityPattern(env, UAV_id, current_position, target_position):
        if target_position is None:
            # 悬停
            mobility_pattern = {'angle': 0, 'phi': 0, 'speed': 0}
            target_position = current_position
        else:
            # 检查是否需要规划避开禁飞区的路径
            nonfly_zones = env.traffic_manager.getNonFlyZones()
            needs_planning = False
            for zone in nonfly_zones:
                if TrafficScheduler.isLineCrossNonFlyZone(current_position, target_position, zone):
                    needs_planning = True
                    break
                    
            if needs_planning:
                # 规划避开禁飞区的路径
                path = TrafficScheduler.planPathAvoidingNonFlyZones(env, current_position, target_position)
                if len(path) > 1:
                    # 使用路径中的第一个航点作为下一个目标
                    next_waypoint = path[1]
                    # 将完整路径保存在环境中以便后续使用
                    TrafficScheduler.addUAVRoute(env, UAV_id, {'position': next_waypoint, 'to_stay_time': 0})
                    for wp in path[2:]:
                        TrafficScheduler.addUAVRoute(env, UAV_id, {'position': wp, 'to_stay_time': 0})
                    
                    # 计算航向角和仰角
                    delta_x = next_waypoint[0] - current_position[0]
                    delta_y = next_waypoint[1] - current_position[1]
                    delta_z = next_waypoint[2] - current_position[2]
                    
                    # 计算 xy 平面的方位角
                    angle = np.arctan2(delta_y, delta_x)
                    
                    # 计算 z 相对于 xy 平面的仰角
                    distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
                    phi = np.arctan2(delta_z, distance_xy)
                    
                    mobility_pattern = {'angle': angle, 'phi': phi}
                    UAV_speed_range = TrafficScheduler.getConfig(env, 'UAV_speed_range')
                    mobility_pattern['speed'] = UAV_speed_range[1]
                else:
                    # 如果无法找到路径，则悬停
                    mobility_pattern = {'angle': 0, 'phi': 0, 'speed': 0}
            else:
                # 直接飞行到目标点
                delta_x = target_position[0] - current_position[0]
                delta_y = target_position[1] - current_position[1]
                delta_z = target_position[2] - current_position[2]
                
                # 计算 xy 平面的方位角
                angle = np.arctan2(delta_y, delta_x)
                
                # 计算 z 相对于 xy 平面的仰角
                distance_xy = np.sqrt(delta_x ** 2 + delta_y ** 2)
                phi = np.arctan2(delta_z, distance_xy)
                
                mobility_pattern = {'angle': angle, 'phi': phi}
                UAV_speed_range = TrafficScheduler.getConfig(env, 'UAV_speed_range')
                mobility_pattern['speed'] = UAV_speed_range[1]
        
        mobility_pattern['target_position'] = target_position
        return mobility_pattern

    @staticmethod
    def getNextPositionOfUav(env, UAV_id):
        route = env.uav_routes.get(UAV_id, [])
        if len(route) == 0:
            return None
        else:
            return copy.deepcopy(route[0]['position'])

    @staticmethod
    def addUAVRoute(env, UAV_id, pos_with_time):
        route = env.uav_routes.get(UAV_id, [])
        route.append(pos_with_time)
        env.uav_routes[UAV_id] = route

    @staticmethod
    def updateRoute(env, UAV_id, stay_time):
        route = env.uav_routes.get(UAV_id, [])
        assert len(route) > 0, f"Route length of {UAV_id} should larger than 0."
        route[0]['to_stay_time'] = max(route[0]['to_stay_time'] - stay_time, 0)
        if route[0]['to_stay_time'] <= 0:
            del route[0]
        env.uav_routes[UAV_id] = route

    @staticmethod
    def getNearestRSUById(env, node_id):
        rsu_infos = env.traffic_manager.getRSUInfos()
        rsu_ids = list(rsu_infos.keys())
        rsu_positions = [rsu_infos[rsu_id]['position'] for rsu_id in rsu_ids]
        node_position = env.traffic_manager.getNodePositionById(node_id)
        if node_position is None:
            return rsu_ids[0]
        distances = np.linalg.norm(np.asarray(rsu_positions) - np.asarray(node_position), axis=1)
        nearest_idx = np.argmin(distances)
        return rsu_ids[nearest_idx]
