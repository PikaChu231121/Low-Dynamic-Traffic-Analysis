import traci
import numpy as np
import random
import pandas as pd
import sumolib
from shapely.geometry import Point, Polygon

class TrafficManager():
    """The traffic manager class. It manages both vehicle traffic and UAV traffic. It also manipulates the positions of the vehicles, UAVs, RSUs, and cloud servers.
    """

    def __init__(self, config_traffic, traci_connection:traci.connection, sumo_network_xml:str=None):
        """Initialize the traffic manager.

        Args:
            config_traffic (dict): The traffic configuration part of the environment configuration.
        """
        self._config_traffic = config_traffic
        self._net = sumolib.net.readNet(sumo_network_xml)
        self._max_n_vehicles = config_traffic.get("max_n_vehicles", 100)
        self._x_range = config_traffic.get("x_range", [0, 1000]) # set in airfogsim_env.py according to used area map
        self._y_range = config_traffic.get("y_range", [0, 1000]) # set in airfogsim_env.py according to used area map
        self._nonfly_zone_coordinates = config_traffic.get("nonfly_zone_coordinates", [])
        self._UAV_z_range = config_traffic.get("UAV_z_range", [100, 200])
        self._UAV_speed_range=config_traffic.get("UAV_speed_range",[20,40])
        self._max_n_UAVs = config_traffic.get("max_n_UAVs", 10)
        self._RSU_positions = config_traffic.get("RSU_positions", [[0,0,0]])
        self._max_n_cloudServers = config_traffic.get("max_n_cloudServers", 1)
        self._arrival_lambda = config_traffic.get("arrival_lambda", 1)
        self._distance_threshold = config_traffic.get("distance_threshold", 1)

        self._traci_connection = traci_connection
        self._current_time = 0.0

        self._vehicle_infos = {} # vehicle_id -> {position, speed, routeId}
        self._UAV_infos = {} # uav_id -> {position, speed, acceleration, angle, phi}
        self._RSU_infos = {} # rsu_id -> {position, id}
        self._cloudServer_infos = {}
        self._new_added_vehicle_ids=[] # The latest(this time slot) added vehicle's id

        self._sumo_route_ids = [] # all route ids in SUMO, further information can be gained by traci_connection.route.getEdges(route_id)
        self._sumo_edges = {} # each edge is a series of lanes in SUMO, edgeId -> [laneId1, laneId2, ...]
        self._sumo_laneIds = [] # all lane ids in SUMO

        self._traffic_interval = config_traffic.get("traffic_interval", 1)
        self._tripinfo = None
        if traci_connection is not None:
            assert traci_connection.simulation.getDeltaT() == self._traffic_interval, "The traffic interval should be the same as the simulation interval."
        else:
            # 从config_traffic的tripinfo中获取tripinfo.csv的路径，读取作为pandas的DataFrame
            tripinfo_path = config_traffic.get("tripinfo", None)
            if tripinfo_path is not None:
                # 只读取columns=['vehicle_id', 'data_timestep', 'vehicle_x', 'vehicle_y', 'vehicle_speed', 'vehicle_angle', 'vehicle_route']
                self._tripinfo = pd.read_csv(tripinfo_path, sep=";", usecols=['vehicle_id', 'data_timestep', 'vehicle_x', 'vehicle_y', 'vehicle_speed', 'vehicle_angle', 'vehicle_route'])
                # dropnan
                self._tripinfo = self._tripinfo.dropna()
            else:
                raise ValueError("The tripinfo path is not set in the config_traffic.")
        
        self._vehicle_id_counter = 0
        self._UAV_id_counter = 0
        self._RSU_id_counter = 0
        self._cloudServer_id_counter = 0
        self._route_id_counter = 0

        self._grid_width = 50
        self._traffic_mode = config_traffic['traffic_mode']

        self._initialize_map_by_grid()
        self._initialize_edges_and_lanes()
        self._update_route_ids()
        self._initialize_RSUs()
        self._initialize_cloudServers()
        self._initialize_UAVs()

    def reset(self, traci_connection = None):
        """Reset the traffic manager.
        """
        self._traci_connection = traci_connection
        self._current_time = 0.0
        self._vehicle_infos = {}
        self._UAV_infos = {}
        self._RSU_infos = {}
        self._cloudServer_infos = {}
        self._new_added_vehicle_ids = []
        self._vehicle_id_counter = 0
        self._UAV_id_counter = 0
        self._RSU_id_counter = 0
        self._cloudServer_id_counter = 0
        self._route_id_counter = 0
        self._initialize_map_by_grid()
        self._initialize_edges_and_lanes()
        self._update_route_ids()
        self._initialize_RSUs()
        self._initialize_cloudServers()
        self._initialize_UAVs()

    def getMapIndexByNodeId(self, node_id):
        # row_idx, col_idx = np.where(self._map_by_grid == node_id)
        for row in range(self._map_by_grid.shape[0]):
            for col in range(self._map_by_grid.shape[1]):
                if node_id in self._map_by_grid[row, col]:
                    return row, col
        return None, None
    
    def getVehicleTrafficInfosByMapIndex(self, row, col):
        row = min(max(0, row), self._map_by_grid.shape[0] - 1)
        col = min(max(0, col), self._map_by_grid.shape[1] - 1)
        vehicle_ids = self._map_by_grid[row, col]
        vehicle_infos = self.getVehicleInfoByIds(vehicle_ids)
        return vehicle_infos
    
    def getMapIndexesByTargetPositionAndRange(self, target_position, range):
        row = int((target_position[1] - self._y_range[0]) / self._grid_width)
        col = int((target_position[0] - self._x_range[0]) / self._grid_width)
        row_range = int(range / self._grid_width)
        col_range = int(range / self._grid_width)
        row_start = max(0, row - row_range)
        row_end = min(self._map_by_grid.shape[0], row + row_range + 1)
        col_start = max(0, col - col_range)
        col_end = min(self._map_by_grid.shape[1], col + col_range + 1)
        return row_start, row_end, col_start, col_end

    @property
    def map_by_grid(self):
        return self._map_by_grid.copy()
    
    @property
    def grid_width(self):
        return self._grid_width

    def _initialize_map_by_grid(self):
        """Initialize the map_by_grid matrix. The matrix is used to store the node ids (as list) in each grid. The grid is defined by the grid width. The matrix is by: row1, col1 = y1, x1; row2, col2 = y2, x2 of position (x, y). 
        """
        row_num = int((self._y_range[1] - self._y_range[0]) / self._grid_width)
        col_num = int((self._x_range[1] - self._x_range[0]) / self._grid_width)
        self._map_by_grid = np.empty((row_num, col_num), dtype=object)
        for i in range(row_num):
            for j in range(col_num):
                self._map_by_grid[i, j] = []

    def getRSUPositions(self):
        """Get the RSU positions.

        Returns:
            list: The RSU positions.
        """
        return self._RSU_positions

    def getNumberOfRSUs(self):
        """Get the number of RSUs.

        Returns:
            int: The number of RSUs.
        """
        return len(self._RSU_infos)
    
    def getNumberOfCloudServers(self):
        """Get the number of cloud servers.

        Returns:
            int: The number of cloud servers.
        """
        return len(self._cloudServer_infos)
    
    def getNumberOfUAVs(self):
        """Get the number of UAVs.

        Returns:
            int: The number of UAVs.
        """
        return len(self._UAV_infos)
    
    def getNumberOfVehicles(self):
        """Get the number of vehicles.

        Returns:
            int: The number of vehicles.
        """
        return len(self._vehicle_infos)

    def _initialize_RSUs(self):
        """Initialize the RSU information.
        """
        for RSU_position in self._RSU_positions:
            RSU_id = "RSU_" + str(self._RSU_id_counter)
            self._RSU_id_counter += 1
            self._RSU_infos[RSU_id] = {"position": RSU_position, "id": RSU_id}
            row = int((RSU_position[1] - self._y_range[0]) / self._grid_width)
            col = int((RSU_position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(RSU_id)

    def _initialize_cloudServers(self):
        """Initialize the cloud server information.
        """
        for _ in range(self._max_n_cloudServers):
            cloudServer_id = "cloudServer_" + str(self._RSU_id_counter)
            self._cloudServer_id_counter += 1
            position = (0, 0, 0)
            self._cloudServer_infos[cloudServer_id] = {"position": position, "id": cloudServer_id}

    def _is_position_in_nonfly_zone(self, position):
        """检查位置是否在禁飞区内"""
        point = Point(position[0], position[1])
        for zone in self._nonfly_zone_coordinates:
            polygon = Polygon(zone)
            if polygon.contains(point):
                return True
        return False

    def _initialize_UAVs(self):
        """Initialize the UAV information with random positions in the given range.
        Ensure UAVs are not initialized inside non-fly zones.
        """
        max_attempts = 50  # 每个UAV的最大尝试次数
        
        for _ in range(self._max_n_UAVs):
            UAV_id = "UAV_" + str(self._UAV_id_counter)
            self._UAV_id_counter += 1
            
            # 尝试生成不在禁飞区的位置
            position = None
            for attempt in range(max_attempts):
                candidate_position = (
                    random.uniform(self._x_range[0], self._x_range[1]), 
                    random.uniform(self._y_range[0], self._y_range[1]), 
                    random.uniform(self._UAV_z_range[0], self._UAV_z_range[1])
                )
                
                if not self._is_position_in_nonfly_zone(candidate_position):
                    position = candidate_position
                    break
            
            if position is None:
                # 如果无法找到合适位置，选择一个远离禁飞区的边缘位置
                position = (
                    self._x_range[0] + 100,  # 距离边界100单位
                    self._y_range[0] + 100, 
                    random.uniform(self._UAV_z_range[0], self._UAV_z_range[1])
                )
            
            self._UAV_infos[UAV_id] = {"position": position}
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(UAV_id)

    def _initialize_edges_and_lanes(self):
        """Initialize the edges and lanes information."""
        if self._net:
            self._sumo_edges = {}
            self._sumo_laneIds = []
            nodes = self._net.getNodes()
            self._sumo_junction_positions = {}
            for node in nodes:
                node_id = node.getID()
                position = (node.getCoord()[0], node.getCoord()[1], 0)
                self._sumo_junction_positions[node_id] = position
            
            # 获取所有的 lane IDs
            for edge in self._net.getEdges():
                for lane in edge.getLanes():
                    lane_id = lane.getID()
                    edge_id = edge.getID()
                    if edge_id not in self._sumo_edges:
                        self._sumo_edges[edge_id] = []
                    self._sumo_edges[edge_id].append(lane_id)
                    self._sumo_laneIds.append(lane_id)

            valid_edges = []
            edges = list(self._sumo_edges.keys())
            self.all_allowed_classes = set()

            for edge in edges:
                lanes = self._sumo_edges[edge]
                for lane_id in lanes:
                    lane = self._net.getLane(lane_id)
                    allowed_classes = lane.getPermissions()
                    self.all_allowed_classes.update(allowed_classes)
                    if len(allowed_classes) == 0 or 'passenger' in allowed_classes:
                        valid_edges.append(edge)
                        break

            self.valid_edges = valid_edges
        

    def _update_route_ids(self):
        """Update the route information generated by SUMO.
        """
        if self._traffic_mode == 'SUMO':
            route_ids = self._traci_connection.route.getIDList()
            self._sumo_route_ids = route_ids

    def _generateRandomRoute(self):
        """Generate a random route id.

        Returns:
            str: The route id.
        """
        route_id = "gen_veh_route_" + str(self._route_id_counter)
        valid_edges = self.valid_edges
        while True:
            try:
                from_edge, to_edge = random.sample(valid_edges, 2) 
                route = traci.simulation.findRoute(from_edge, to_edge)
                while len(route.edges) == 0:
                    from_edge, to_edge = random.sample(valid_edges, 2)
                    route = traci.simulation.findRoute(from_edge, to_edge)
                break
            except traci.exceptions.TraCIException as e:
                pass
            
        self._traci_connection.route.add(route_id, route.edges)
        self._route_id_counter += 1
        return route_id
    
    def updateVehicleMobilityPatterns(self, vehicle_mobility_patterns):
        """Update the vehicle mobility patterns.

        Args:
            vehicle_mobility_patterns (dict): The vehicle mobility patterns. The key is vehicle id, and the value is the mobility pattern={angle, speed}
        """
        for vehicle_id, mobility_pattern in vehicle_mobility_patterns.items():
            self._traci_connection.vehicle.setSpeed(vehicle_id, mobility_pattern["speed"])

    def _updateUAVMobilityPatternById(self, UAV_id, mobility_pattern):
        """Update the UAV mobility pattern by the UAV id.

        Args:
            UAV_id (str): The UAV id.
            mobility_pattern (dict): The mobility pattern={angle, phi, speed}
        """
        assert UAV_id in self._UAV_infos, "The UAV id should be in the UAV information."
        self._UAV_infos[UAV_id]["speed"] = mobility_pattern["speed"]
        self._UAV_infos[UAV_id]["angle"] = mobility_pattern["angle"]
        self._UAV_infos[UAV_id]["phi"] = mobility_pattern["phi"]

    def updateUAVMobilityPatterns(self, UAV_mobility_patterns):
        """Update the UAV mobility patterns.

        Args:
            UAV_mobility_patterns (dict): The UAV mobility patterns. The key is UAV id, and the value is the mobility pattern={angle, phi, speed}
        """
        for UAV_id, mobility_pattern in UAV_mobility_patterns.items():
            self._updateUAVMobilityPatternById(UAV_id, mobility_pattern)

    def updateCurrentTime(self):
        """Update the current time.

        Returns:
            float: The updated current time.
        """
        if self._traffic_mode == 'SUMO':
            return self._traci_connection.simulation.getTime()
        else:
            # 把self._tripinfo中current_time之前的数据删除
            # self._tripinfo = self._tripinfo[self._tripinfo['data_timestep']>=self._current_time]
            tmp_time = self._current_time + self._traffic_interval
            # 保证tmp_time mod self._traffic_interval == 0
            tmp_time = round(tmp_time / self._traffic_interval) 
            tmp_time = tmp_time * self._traffic_interval
            return tmp_time

    def getVehicleIDsList(self):
        """Get the vehicle ids list.

        Returns:
            list: The vehicle ids list.
        """
        if self._traffic_mode == 'SUMO':
            return self._traci_connection.vehicle.getIDList()
        else:
            # 根据当前的时隙，从tripinfo中获取当前时隙的车辆信息
            current_time = self._current_time
            # tripinfo是pd.DataFrame，可以直接使用pandas的查询功能,date_timestep在current_time-traffic_interval到current_time之间的车辆
            vehicle_ids = self._tripinfo[(self._tripinfo['data_timestep']>current_time-self._traffic_interval) & (self._tripinfo['data_timestep']<=current_time)]['vehicle_id'].tolist()
            return vehicle_ids
        
    def getVehicleInfoByIds(self, vehicle_ids):
        # {"position": position3d, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
        if self._traffic_mode == 'SUMO':
            vehicle_infos = {}
            for vehicle_id in vehicle_ids:
                position = self._traci_connection.vehicle.getPosition(vehicle_id)
                speed = self._traci_connection.vehicle.getSpeed(vehicle_id)
                acceleration = self._traci_connection.vehicle.getAcceleration(vehicle_id)
                angle = self._traci_connection.vehicle.getAngle(vehicle_id)
                route_id = self._traci_connection.vehicle.getRouteID(vehicle_id)
                position3d = (position[0], position[1], 0)
                vehicle_infos[vehicle_id] = {"position": position3d, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
            return vehicle_infos
        else:
            # 从pd中批量获取车辆信息
            cur_time_trip_info = self._tripinfo[(self._tripinfo['data_timestep']>self._current_time-self._traffic_interval) & (self._tripinfo['data_timestep']<=self._current_time)]
            pd_vehicle_infos = cur_time_trip_info[cur_time_trip_info['vehicle_id'].isin(vehicle_ids)]
            vehicle_infos = {}
            for idx, vehicle_info in pd_vehicle_infos.iterrows():
                position = (vehicle_info['vehicle_x'], vehicle_info['vehicle_y'], 0)
                speed = vehicle_info['vehicle_speed']
                acceleration = 0
                angle = vehicle_info['vehicle_angle']
                route_id = vehicle_info['vehicle_route']
                vehicle_id = vehicle_info['vehicle_id']
                vehicle_infos[vehicle_id] = {"position": position, "speed": speed, "acceleration": acceleration, "angle": angle, "routeId": route_id, 'id': vehicle_id}
            return vehicle_infos

    def stepSimulation(self):
        """Step the simulation for one step. Generate vehicles according to Poisson distribution, limit the number of vehicles, and update the route ids.
        """
        if self._traffic_mode == 'SUMO':
            to_generate_vehicles = int(np.random.poisson(self._arrival_lambda*self._traffic_interval))
            current_n_vehicles = self._traci_connection.vehicle.getIDCount()
            to_generate_vehicles = min(to_generate_vehicles, self._max_n_vehicles - current_n_vehicles)
            self._new_added_vehicle_ids = []  # Clear the list in each step.
            if to_generate_vehicles > 0 :
                for _ in range(to_generate_vehicles):
                    vehicle_id = "vehicle_" + str(self._vehicle_id_counter)
                    self._new_added_vehicle_ids.append(vehicle_id)
                    self._vehicle_id_counter += 1
                    route_id = self._generateRandomRoute()
                    self._traci_connection.vehicle.add(vehicle_id, route_id)
            self._traci_connection.simulationStep()
            # vehicles will be updated by sumo. (Vehicles which are out of map will be cleared automatically by sumo)
            vehicle_ids = self.getVehicleIDsList()
        else:
            # 从tripinfo中获取当前时间的车辆信息
            vehicle_ids = self.getVehicleIDsList()
        self._current_time = self.updateCurrentTime()
        self._vehicle_infos = self.getVehicleInfoByIds(vehicle_ids)

        for UAV_id in self._UAV_infos:
            org_position = self._UAV_infos[UAV_id]["position"]
            speed = self._UAV_infos[UAV_id].get("speed", 0)
            last_speed = self._UAV_infos[UAV_id].get("last_speed", 0)
            acceleration = (last_speed - speed) / self._traffic_interval
            self._UAV_infos[UAV_id]["acceleration"] = acceleration
            self._UAV_infos[UAV_id]["last_speed"] = self._UAV_infos[UAV_id].get("speed", 0)
            angle = self._UAV_infos[UAV_id].get("angle", 0)
            phi = self._UAV_infos[UAV_id].get("phi", 0)
            
            # new position of UAV need to be uodated by hand
            new_position = (org_position[0] + speed * np.cos(angle) * np.cos(phi) * self._traffic_interval, 
                            org_position[1] + speed * np.sin(angle) * np.cos(phi) * self._traffic_interval, 
                            org_position[2] + speed * np.sin(phi) * self._traffic_interval)
            new_position = [float(i) for i in new_position]
            
            # 检查新位置是否在禁飞区内，如果是则调整位置
            new_position = self._adjust_position_if_in_nonfly_zone(new_position, org_position)
            
            self._UAV_infos[UAV_id] = {"position": new_position, "speed": speed, "last_speed": speed, 
                                       "angle": angle, "phi": phi, "acceleration": acceleration}
        
        # 检查并移动所有在禁飞区内的UAV
        self._rescue_uavs_in_nonfly_zones()
        
        self._update_route_ids()
        self._update_map_by_grid()
        
    def _adjust_position_if_in_nonfly_zone(self, new_position, org_position):
        """如果新位置在禁飞区内，调整位置以避开禁飞区"""
        # 检查新位置是否在禁飞区内
        for nonfly_zone in self._nonfly_zone_coordinates:
            polygon = Polygon(nonfly_zone)
            point = Point(new_position[0], new_position[1])
            
            if polygon.contains(point):
                # 位置在禁飞区内，需要调整
                # 计算当前航向角
                delta_x = new_position[0] - org_position[0]
                delta_y = new_position[1] - org_position[1]
                angle = np.arctan2(delta_y, delta_x)
                
                # 尝试不同的方向直到找到一个不在禁飞区内的位置
                for angle_offset in [0, np.pi/4, -np.pi/4, np.pi/2, -np.pi/2, 3*np.pi/4, -3*np.pi/4, np.pi]:
                    test_angle = angle + angle_offset
                    distance = np.sqrt(delta_x**2 + delta_y**2)
                    
                    # 计算新的可能位置
                    test_position = [
                        org_position[0] + distance * np.cos(test_angle),
                        org_position[1] + distance * np.sin(test_angle),
                        new_position[2]  # 保持同样的高度
                    ]
                    
                    # 检查测试位置是否在禁飞区外
                    test_point = Point(test_position[0], test_position[1])
                    if not polygon.contains(test_point):
                        return test_position
                
                # 如果所有方向都不行，则垂直上升
                return [org_position[0], org_position[1], new_position[2] + 50]
                
        # 如果新位置不在任何禁飞区内，则返回原始新位置
        return new_position

    def _rescue_uavs_in_nonfly_zones(self):
        """检查并移动所有在禁飞区内的UAV"""
        for UAV_id, UAV_info in self._UAV_infos.items():
            position = UAV_info["position"]
            
            if self._is_position_in_nonfly_zone(position):
                # 无人机在禁飞区内，需要移动到安全位置
                # 找到最近的禁飞区边缘并移出
                new_position = self._find_nearest_safe_position(position)
                
                # 更新无人机位置，保留其他属性
                self._UAV_infos[UAV_id]["position"] = new_position
                
                print(f"已将无人机 {UAV_id} 从禁飞区移至安全位置: {new_position}")
    
    def _find_nearest_safe_position(self, current_position):
        """找到最近的安全位置（不在禁飞区内）"""
        # 首先尝试原位置增加高度
        test_position = (current_position[0], current_position[1], current_position[2] + 50)
        if not self._is_position_in_nonfly_zone(test_position):
            return test_position
            
        # 计算到禁飞区边缘的方向和距离
        directions = [(1, 0), (-1, 0), (0, 1), (0, -1), 
                      (1, 1), (-1, -1), (1, -1), (-1, 1)]
        
        for distance in [50, 100, 150, 200]:  # 逐渐增加距离
            for dx, dy in directions:
                test_position = (
                    current_position[0] + dx * distance, 
                    current_position[1] + dy * distance,
                    current_position[2]
                )
                
                if not self._is_position_in_nonfly_zone(test_position):
                    return test_position
        
        # 如果上述方法都失败，放在地图安全位置
        return (
            self._x_range[0] + 200,
            self._y_range[0] + 200,
            current_position[2]
        )

    def _update_map_by_grid(self):
        self._map_by_grid = np.empty((self._map_by_grid.shape[0], self._map_by_grid.shape[1]), dtype=object)
        for i in range(self._map_by_grid.shape[0]):
            for j in range(self._map_by_grid.shape[1]):
                self._map_by_grid[i, j] = []
        for vehicle_id, vehicle_info in self._vehicle_infos.items():
            position = vehicle_info["position"]
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(vehicle_id)
        for UAV_id, UAV_info in self._UAV_infos.items():
            position = UAV_info["position"]
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(UAV_id)
        for RSU_id, RSU_info in self._RSU_infos.items():
            position = RSU_info["position"]
            row = int((position[1] - self._y_range[0]) / self._grid_width)
            col = int((position[0] - self._x_range[0]) / self._grid_width)
            if row >= 0 and row < self._map_by_grid.shape[0] and col >= 0 and col < self._map_by_grid.shape[1]:
                self._map_by_grid[row, col].append(RSU_id)

    def getVehicleTrafficInfos(self):
        """Get the vehicle traffics at the given simulation time.

        Returns:
            dict: The vehicle traffics, including the vehicle id, position, speed, angle, acceleration, and current routeId.
        """
        return self._vehicle_infos
    
    def getUAVTrafficInfos(self):
        """Get the UAV traffics at the given simulation time. The trajectory of the UAVs is controlled by their missions

        Returns:
            dict: The UAV traffics, including the UAV id, position, acceleration, speed, angle, and phi.
        """
        return self._UAV_infos
    
    def getRSUInfos(self):
        """Get the RSU information.

        Returns:
            dict: The RSU information, including the RSU id and position
        """
        return self._RSU_infos
    
    def getCloudServerInfos(self):
        """Get the cloud server information.

        Returns:
            dict: The cloud server information, including the cloud server id and position.
        """
        return self._cloudServer_infos

    def getNewVehicleIds(self):
        """Get vehicle ids which is added in latest timeslot.

        Returns:
            list: The Id list of vehicles.
        """
        return self._new_added_vehicle_ids
    
    def getCurrentTime(self):
        """Get the current simulation time.

        Returns:
            float: The current simulation time (in seconds).
        """
        # return self._traci_connection.simulation.getTime()
        return self._current_time

    def removeUAV(self,UAV_id):
        assert UAV_id in self._UAV_infos.keys(),'UAV_id not in _UAV_infos'
        del self._UAV_infos[UAV_id]

    def checkIsRemovingByUAVId(self,UAV_id):
        UAV_info=self._UAV_infos[UAV_id]
        assert UAV_id in self._UAV_infos.keys(), 'UAV_id not in _UAV_infos'
        return UAV_info['speed']>0

    def getConfig(self,name):
        return self._config_traffic.get(name,None)
    
    def getNodePositionById(self, id):
        if id in self._vehicle_infos:
            return self._vehicle_infos[id]["position"]
        elif id in self._UAV_infos:
            return self._UAV_infos[id]["position"]
        elif id in self._RSU_infos:
            return self._RSU_infos[id]["position"]
        elif id in self._cloudServer_infos:
            return self._cloudServer_infos[id]["position"]
        return None
    
    def getAllJunctionPositions(self):
        return list(self._sumo_junction_positions.values())
    
    def getNonFlyZones(self):
        return self._nonfly_zone_coordinates.copy()