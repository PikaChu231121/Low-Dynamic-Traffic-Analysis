import os
import yaml
import numpy as np
import xml.etree.ElementTree as ET
import math
from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap


def _load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class ContextExtractor:
    def __init__(self, simulator, config_path):
        """
        初始化上下文提取器
        
        Args:
            simulator: AirFogSim模拟器实例
            config_path: 配置文件路径
        """
        self.simulator = simulator
        self.config = _load_config(config_path)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.sumo_net_file = os.path.join(project_root, 
                                         self.config['sumo']['sumo_net'])
        self.context_data = {}
        # 添加一个字典用于存储多个时间点的交通密度数据
        self.traffic_density_history = {}

    def extract_all_context(self):
        """提取所有前置上下文信息"""
        self.extract_traffic_topology()
        self.extract_nonfly_zones()
        self.extract_computation_capacity()
        self.extract_mission_characteristics()
    
    def extract_traffic_topology(self):
        """从SUMO网络文件中提取交通拓扑信息"""
        try:
            tree = ET.parse(self.sumo_net_file)
            root = tree.getroot()
            
            # 提取交叉路口信息
            junctions = []
            for junction in root.findall('.//junction'):
                if junction.get('type') not in ['internal', 'dead_end']:
                    junctions.append({
                        'id': junction.get('id'),
                        'x': float(junction.get('x')),
                        'y': float(junction.get('y')),
                        'type': junction.get('type')
                    })
            
            # 提取道路信息
            edges = []
            for edge in root.findall('.//edge'):
                if edge.get('function') != 'internal':
                    edge_data = {
                        'id': edge.get('id'),
                        'from': edge.get('from'),
                        'to': edge.get('to'),
                        'priority': edge.get('priority'),
                        'lanes': []
                    }
                    
                    # 提取车道信息
                    for lane in edge.findall('.//lane'):
                        edge_data['lanes'].append({
                            'id': lane.get('id'),
                            'speed': float(lane.get('speed')),
                            'length': float(lane.get('length')),
                            'shape': lane.get('shape')
                        })
                    
                    edges.append(edge_data)
            
            # 识别主要交通走廊和拥堵点
            self.context_data['traffic_topology'] = {
                'junctions': junctions,
                'edges': edges,
                'junction_count': len(junctions),
                'edge_count': len(edges)
            }
            
            # 识别关键交叉路口（按连接边的数量排序）
            junction_connections = {}
            for edge in edges:
                from_junction = edge.get('from')
                to_junction = edge.get('to')
                
                if from_junction not in junction_connections:
                    junction_connections[from_junction] = 0
                junction_connections[from_junction] += 1
                
                if to_junction not in junction_connections:
                    junction_connections[to_junction] = 0
                junction_connections[to_junction] += 1
            
            # 找出连接数最多的前5个交叉路口作为关键交叉路口
            key_junctions = sorted(junction_connections.items(), 
                                  key=lambda x: x[1], reverse=True)[:5]
            
            self.context_data['key_junctions'] = key_junctions
            
            print(f"成功提取交通拓扑信息: {len(junctions)}个交叉路口, {len(edges)}条道路")
        except Exception as e:
            print(f"提取交通拓扑信息失败: {e}")
            self.context_data['traffic_topology'] = {
                'error': str(e)
            }
    
    def extract_nonfly_zones(self):
        """提取禁飞区信息"""
        nonfly_zones = self.config['traffic'].get('nonfly_zone_coordinates', [])
        
        # 转换为多边形并计算面积
        nonfly_polygons = []
        for zone in nonfly_zones:
            try:
                poly = Polygon(zone)
                nonfly_polygons.append({
                    'coordinates': zone,
                    'area': poly.area,
                    'centroid': (poly.centroid.x, poly.centroid.y)
                })
            except Exception as e:
                print(f"处理禁飞区多边形时出错: {e}")
        
        # 计算总禁飞面积与模拟区域的比例
        if nonfly_polygons:
            # 假设模拟区域是一个由最大和最小坐标定义的矩形
            all_points = [point for zone in nonfly_zones for point in zone]
            if all_points:
                x_coords = [p[0] for p in all_points]
                y_coords = [p[1] for p in all_points]
                
                # 估计模拟区域范围
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)
                
                # 确保有一个合理的边界
                buffer = 1000  # 1000单位的缓冲区
                simulation_area = (max_x - min_x + 2*buffer) * (max_y - min_y + 2*buffer)
                
                # 计算禁飞区总面积
                total_nonfly_area = sum(poly['area'] for poly in nonfly_polygons)
                
                # 计算比例
                nonfly_ratio = total_nonfly_area / simulation_area
            else:
                nonfly_ratio = 0
        else:
            nonfly_ratio = 0
            
        self.context_data['nonfly_zones'] = {
            'zones': nonfly_polygons,
            'count': len(nonfly_polygons),
            'total_area': sum(poly['area'] for poly in nonfly_polygons) if nonfly_polygons else 0,
            'area_ratio': nonfly_ratio
        }
        
        print(f"成功提取禁飞区信息: {len(nonfly_polygons)}个禁飞区, 总面积占比: {nonfly_ratio:.2%}")
    
    def extract_computation_capacity(self):
        """提取计算能力信息"""
        fog_profile = self.config['fog_profile']
        
        self.context_data['computation_capacity'] = {
            'vehicle': fog_profile.get('vehicle', {}),
            'uav': fog_profile.get('uav', {}),
            'rsu': fog_profile.get('rsu', {}),
            'cloud': fog_profile.get('cloud', {}),
            'vehicle_count': self.config['simulation'].get('vehicle_count', 0),
            'uav_count': self.config['traffic'].get('max_n_UAVs', 0),
            'rsu_count': len(self.config['traffic'].get('RSU_positions', [])),
            'cloud_count': self.config['traffic'].get('max_n_cloudServers', 0)
        }
        
        # 计算总计算能力
        total_cpu = (
            fog_profile.get('vehicle', {}).get('cpu', 0) * self.config['simulation'].get('vehicle_count', 0) +
            fog_profile.get('uav', {}).get('cpu', 0) * self.config['traffic'].get('max_n_UAVs', 0) +
            fog_profile.get('rsu', {}).get('cpu', 0) * len(self.config['traffic'].get('RSU_positions', [])) +
            fog_profile.get('cloud', {}).get('cpu', 0) * self.config['traffic'].get('max_n_cloudServers', 0)
        )
        
        self.context_data['computation_capacity']['total_cpu'] = total_cpu
        print(f"成功提取计算能力信息, 总CPU容量: {total_cpu}")
    
    def extract_traffic_density_areas(self, simulation_time=None):
        """识别高交通流密度区域，分别计算车辆和无人机的密度"""
        if not hasattr(self, 'simulator') or self.simulator is None:
            density_data = {
                'note': "模拟器未初始化"
            }
            if simulation_time is not None:
                self.traffic_density_history[simulation_time] = density_data
            return

        try:
            # 获取车辆和无人机位置
            vehicle_positions = []
            uav_positions = []
            if hasattr(self.simulator, 'traffic_manager'):
                vehicle_infos = self.simulator.traffic_manager.getVehicleTrafficInfos()
                for info in vehicle_infos.values():
                    pos = info.get('position')
                    if pos:
                        vehicle_positions.append((pos[0], pos[1]))

                uav_infos = self.simulator.traffic_manager.getUAVTrafficInfos()
                for info in uav_infos.values():
                    pos = info.get('position')
                    if pos:
                        uav_positions.append((pos[0], pos[1]))

            # 如果没有交通数据，返回空结果
            if not vehicle_positions and not uav_positions:
                density_data = {
                    'note': "无交通数据可用",
                    'time': simulation_time
                }
                if simulation_time is not None:
                    self.traffic_density_history[simulation_time] = density_data
                return

            # 使用网格方法分别计算车辆和无人机的密度
            all_positions = vehicle_positions + uav_positions
            if all_positions:
                x_coords = [p[0] for p in all_positions]
                y_coords = [p[1] for p in all_positions]
                min_x, max_x = min(x_coords), max(x_coords)
                min_y, max_y = min(y_coords), max(y_coords)

                buffer = 500
                min_x -= buffer
                max_x += buffer
                min_y -= buffer
                max_y += buffer

                grid_size = 500
                grid_width = int((max_x - min_x) / grid_size) + 1
                grid_height = int((max_y - min_y) / grid_size) + 1

                vehicle_density_grid = np.zeros((grid_height, grid_width))
                uav_density_grid = np.zeros((grid_height, grid_width))

                for pos in vehicle_positions:
                    grid_x = int((pos[0] - min_x) / grid_size)
                    grid_y = int((pos[1] - min_y) / grid_size)
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        vehicle_density_grid[grid_y, grid_x] += 1

                for pos in uav_positions:
                    grid_x = int((pos[0] - min_x) / grid_size)
                    grid_y = int((pos[1] - min_y) / grid_size)
                    if 0 <= grid_x < grid_width and 0 <= grid_y < grid_height:
                        uav_density_grid[grid_y, grid_x] += 1

                # 将计数值转换为单位面积的密度
                conversion_factor = 1000000
                grid_area = grid_size * grid_size
                vehicle_density_grid = vehicle_density_grid / grid_area * conversion_factor
                uav_density_grid = uav_density_grid / grid_area * conversion_factor

                avg_vehicle_density = np.mean(vehicle_density_grid)
                avg_uav_density = np.mean(uav_density_grid)

                density_data = {
                    'vehicle_density_grid': vehicle_density_grid.tolist(),
                    'uav_density_grid': uav_density_grid.tolist(),
                    'avg_vehicle_density': float(avg_vehicle_density),
                    'avg_uav_density': float(avg_uav_density),
                    'vehicle_count': len(vehicle_positions),
                    'uav_count': len(uav_positions),
                    'time': simulation_time
                }

                if simulation_time is not None:
                    self.traffic_density_history[simulation_time] = density_data

                print(f"成功提取交通密度信息[时间点 {simulation_time}]: 车辆密度平均值 {avg_vehicle_density:.2f}, 无人机密度平均值 {avg_uav_density:.2f}")
        except Exception as e:
            print(f"提取交通密度信息失败: {e}")
            density_data = {
                'error': str(e),
                'time': simulation_time
            }
            if simulation_time is not None:
                self.traffic_density_history[simulation_time] = density_data

    def extract_mission_characteristics(self):
        """提取任务特征信息"""
        task_config = self.config['task']
        mission_config = self.config.get('mission', {})
        
        self.context_data['mission_characteristics'] = {
            'task_generation': {
                'model': task_config.get('task_generation_model', ''),
                'lambda': task_config.get('task_generation_kwargs', {}).get('lambda', 0)
            },
            'task_deadline': {
                'min': task_config.get('task_min_deadline', 0),
                'max': task_config.get('task_max_deadline', 0),
                'hard_ddl': task_config.get('hard_ddl', 0)
            },
            'task_size': {
                'min': task_config.get('task_min_size', 0),
                'max': task_config.get('task_max_size', 0)
            },
            'mission': {
                'UAV_height': mission_config.get('UAV_height', 0),
                'TTL_range': mission_config.get('TTL_range', [0, 0]),
                'distance_threshold': mission_config.get('distance_threshold', 0)
            }
        }
        print("成功提取任务特征信息")
    
    def visualize_context(self, output_path=None):
        """可视化上下文信息，增强交叉路口和关键交叉路口的显示"""
        fig, ax = plt.subplots(figsize=(16, 14))
        
        # 绘制禁飞区
        nonfly_zones = self.config['traffic'].get('nonfly_zone_coordinates', [])
        if nonfly_zones:
            # 只在有区域时添加标签，避免重复
            first_zone = True
            for zone in nonfly_zones:
                label = 'No-Fly Zone' if first_zone else None
                poly = patches.Polygon(zone, closed=True, fill=True, alpha=0.4, 
                                     color='red', label=label)
                ax.add_patch(poly)
                first_zone = False
        
        # 绘制RSU位置
        rsu_positions = self.config['traffic'].get('RSU_positions', [])
        for pos in rsu_positions:
            ax.scatter(pos[0], pos[1], c='blue', s=120, marker='^', 
                      label='RSU' if pos == rsu_positions[0] else "")
        
        # 绘制道路网络
        if 'traffic_topology' in self.context_data:
            edges = self.context_data['traffic_topology'].get('edges', [])
            for edge in edges:  
                for lane in edge.get('lanes', []):
                    shape = lane.get('shape', '')
                    if shape:
                        points = []
                        for point_str in shape.split():
                            x, y = map(float, point_str.split(','))
                            points.append((x, y))
                        
                        if len(points) >= 2:
                            xs, ys = zip(*points)
                            ax.plot(xs, ys, 'k-', linewidth=0.3, alpha=0.6)
        
        # 绘制所有交叉路口
        if 'traffic_topology' in self.context_data:
            junctions = self.context_data['traffic_topology'].get('junctions', [])
            if junctions:
                junction_x = [j['x'] for j in junctions]
                junction_y = [j['y'] for j in junctions]
                ax.scatter(junction_x, junction_y, c='grey', s=30, alpha=0.7, 
                          marker='o', label='Junction')
        
        # 突出显示关键交叉路口
        if 'key_junctions' in self.context_data:
            key_junctions = self.context_data['key_junctions']
            key_junction_ids = [j[0] for j in key_junctions]
            
            # 查找这些关键交叉路口的坐标
            if 'traffic_topology' in self.context_data:
                junctions = self.context_data['traffic_topology'].get('junctions', [])
                for junction in junctions:
                    if junction['id'] in key_junction_ids:
                        # 绘制关键交叉路口
                        ax.scatter(junction['x'], junction['y'], c='green', s=150, 
                                  marker='*', label='Key Junction' if junction == junctions[0] else "")
                        
                        # 在关键交叉路口旁添加文本标签
                        ax.text(junction['x']+20, junction['y']+20, 
                               f"ID: {junction['id']}", 
                               fontsize=8, color='green', weight='bold')
        
        # 添加交通密度热图（如果有数据）
        if 'traffic_density' in self.context_data and 'grid_data' in self.context_data['traffic_density']:
            grid_data = self.context_data['traffic_density']['grid_data']
            
            try:
                min_x = grid_data['min_x']
                min_y = grid_data['min_y']
                grid_size = grid_data['grid_size']
                density_grid = np.array(grid_data['density_grid'])
                
                # 只在有数据且非零的情况下绘制热图
                if density_grid.size > 0 and np.max(density_grid) > 0:
                    # 创建自定义颜色映射，透明度从0.1到0.7
                    cmap = LinearSegmentedColormap.from_list('density_cmap', 
                                                          [(0, (1, 1, 0, 0.1)),     # 黄色，低密度，半透明
                                                           (0.5, (1, 0.5, 0, 0.4)),  # 橙色，中密度
                                                           (1, (1, 0, 0, 0.7))])    # 红色，高密度
                                                           
                    # 计算热图位置
                    extent = [
                        min_x, 
                        min_x + density_grid.shape[1] * grid_size,
                        min_y, 
                        min_y + density_grid.shape[0] * grid_size
                    ]
                    
                    # 使用imshow绘制热图
                    im = ax.imshow(density_grid, extent=extent, origin='lower', 
                                  cmap=cmap, interpolation='bilinear', alpha=0.6)
                    
                    # 添加颜色条
                    cbar = plt.colorbar(im, ax=ax)
                    cbar.set_label('Traffic Density')
                    
                    # 标记高密度区域
                    high_density_areas = self.context_data['traffic_density'].get('high_density_areas', [])
                    if high_density_areas:
                        for i, area in enumerate(high_density_areas):
                            rect = patches.Rectangle(
                                (area['bounds'][0], area['bounds'][1]),
                                area['bounds'][2] - area['bounds'][0],
                                area['bounds'][3] - area['bounds'][1],
                                linewidth=2, edgecolor='red', facecolor='none',
                                label='High Density Area' if i == 0 else None
                            )
                            ax.add_patch(rect)
            except Exception as e:
                print(f"绘制热图时出错: {e}")

        # 上下翻转图像
        ax.invert_yaxis()

        # 设置图例和标题
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize=10)
        
        ax.set_title('Simulation Context Visualization', fontsize=16)
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.grid(True, alpha=0.3)
        
        # 保存或显示图表
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"上下文可视化结果已保存到 {output_path}")
        else:
            plt.tight_layout()
            plt.show()

    def export_context_data(self, output_path):
        """将上下文数据导出为JSON文件，包含分开的车辆和无人机密度"""
        import json

        def make_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                               np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            else:
                return obj

        export_data = make_serializable(self.context_data.copy())
        export_data['traffic_density_history'] = make_serializable(self.traffic_density_history)

        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=4)

        print(f"上下文数据已导出到 {output_path}")

    def export_context_data_summary(self, output_path):
        """导出精简版的上下文数据，仅包含AI预测所需的关键信息"""
        import json

        def make_serializable(obj):
            if isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, 
                               np.int64, np.uint8, np.uint16, np.uint32, np.uint64)):
                return int(obj)
            elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_serializable(i) for i in obj]
            else:
                return obj

        # 创建精简版数据结构
        summary_data = {
            "simulation_info": {
                "vehicle_count": self.config['simulation'].get('vehicle_count', 0),
                "uav_count": self.config['traffic'].get('max_n_UAVs', 0),
                "rsu_count": len(self.config['traffic'].get('RSU_positions', [])),
                "cloud_count": self.config['traffic'].get('max_n_cloudServers', 0)
            },
            "traffic_topology": {
                "junction_count": self.context_data.get('traffic_topology', {}).get('junction_count', 0),
                "edge_count": self.context_data.get('traffic_topology', {}).get('edge_count', 0),
            },
            "computation_capacity": {
                "total_cpu": self.context_data.get('computation_capacity', {}).get('total_cpu', 0),
                "vehicle_cpu": self.context_data.get('computation_capacity', {}).get('vehicle', {}).get('cpu', 0),
                "uav_cpu": self.context_data.get('computation_capacity', {}).get('uav', {}).get('cpu', 0),
                "rsu_cpu": self.context_data.get('computation_capacity', {}).get('rsu', {}).get('cpu', 0),
                "cloud_cpu": self.context_data.get('computation_capacity', {}).get('cloud', {}).get('cpu', 0)
            },
            "mission_characteristics": self.context_data.get('mission_characteristics', {})
        }

        # 添加关键交叉路口信息（包括坐标）
        key_junctions_with_coords = []
        key_junction_ids = [j[0] for j in self.context_data.get('key_junctions', [])[:5]]  # 获取前5个关键交叉路口ID

        if 'traffic_topology' in self.context_data and 'junctions' in self.context_data['traffic_topology']:
            junctions = self.context_data['traffic_topology']['junctions']
            for junction in junctions:
                if junction['id'] in key_junction_ids:
                    idx = key_junction_ids.index(junction['id'])
                    connections = self.context_data['key_junctions'][idx][1]  # 获取连接数
                    key_junctions_with_coords.append({
                        'id': junction['id'],
                        'x': junction['x'],
                        'y': junction['y'],
                        'connections': connections,
                        'type': junction.get('type', 'unknown')
                    })

        summary_data["traffic_topology"]["key_junctions"] = key_junctions_with_coords

        # 添加禁飞区信息（包括坐标）
        nonfly_zones_with_coords = []
        if 'nonfly_zones' in self.context_data and 'zones' in self.context_data['nonfly_zones']:
            zones = self.context_data['nonfly_zones']['zones']
            for zone in zones:
                nonfly_zones_with_coords.append({
                    'centroid': zone.get('centroid', (0, 0)),
                    'area': zone.get('area', 0),
                    'coordinates': zone.get('coordinates', [])[:4]  # 只保留前几个坐标点，避免数据过大
                })

        summary_data["nonfly_zones"] = {
            "count": self.context_data.get('nonfly_zones', {}).get('count', 0),
            "total_area": self.context_data.get('nonfly_zones', {}).get('total_area', 0),
            "area_ratio": self.context_data.get('nonfly_zones', {}).get('area_ratio', 0),
            "zones": nonfly_zones_with_coords
        }

        # 添加RSU位置信息
        rsu_positions = self.config['traffic'].get('RSU_positions', [])
        summary_data["rsu_positions"] = rsu_positions

        # 添加精简版交通密度历史数据
        density_summary = {}
        for time_point, data in self.traffic_density_history.items():
            if 'vehicle_density_grid' in data and 'uav_density_grid' in data:
                # 提取高密度网格信息
                vehicle_density_grid = np.array(data['vehicle_density_grid'])
                uav_density_grid = np.array(data['uav_density_grid'])

                min_x, min_y = 0, 0  # 假设网格的左下角坐标
                grid_size = 500

                high_density_vehicle = np.argwhere(vehicle_density_grid > np.mean(vehicle_density_grid) + np.std(vehicle_density_grid))
                high_density_uav = np.argwhere(uav_density_grid > np.mean(uav_density_grid) + np.std(uav_density_grid))

                vehicle_high_density_info = [
                    {
                        "coordinates": [
                            min_x + grid_size * y,  # 转换为具体的X坐标
                            min_y + grid_size * x   # 转换为具体的Y坐标
                        ],
                        "density": float(vehicle_density_grid[x, y])
                    }
                    for x, y in high_density_vehicle
                ]
                uav_high_density_info = [
                    {
                        "coordinates": [
                            min_x + grid_size * y,  # 转换为具体的X坐标
                            min_y + grid_size * x   # 转换为具体的Y坐标
                        ],
                        "density": float(uav_density_grid[x, y])
                    }
                    for x, y in high_density_uav
                ]

                density_summary[time_point] = {
                    "time": data.get('time'),
                    "vehicle_count": data.get('vehicle_count', 0),
                    "uav_count": data.get('uav_count', 0),
                    "avg_vehicle_density": data.get('avg_vehicle_density', 0),
                    "avg_uav_density": data.get('avg_uav_density', 0),
                    "high_density_vehicle": vehicle_high_density_info[:5],  # 只保留前5个高密度区域
                    "high_density_uav": uav_high_density_info[:5]  # 只保留前5个高密度区域
                }

        summary_data["traffic_density_timeline"] = density_summary

        # 导出精简版数据
        with open(output_path, 'w') as f:
            json.dump(make_serializable(summary_data), f, indent=2)

        print(f"精简版上下文数据已导出到 {output_path}")

    def visualize_density_history(self, output_path=None, max_plots=6):
        """可视化多个时间点的交通密度热力图，分别展示车辆和无人机的密度"""
        if not self.traffic_density_history:
            print("没有可用的交通密度历史数据进行可视化")
            return

        time_points = sorted(self.traffic_density_history.keys())
        if len(time_points) > max_plots:
            indices = np.linspace(0, len(time_points) - 1, max_plots, dtype=int)
            selected_time_points = [time_points[i] for i in indices]
        else:
            selected_time_points = time_points

        n_plots = len(selected_time_points)
        n_cols = 2  # 每行显示车辆和无人机的密度图
        n_rows = n_plots

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 5 * n_rows))
        if n_rows == 1:
            axes = np.array([axes])  # 确保axes是数组
        axes = axes.reshape(n_rows, n_cols)

        for i, t in enumerate(selected_time_points):
            data = self.traffic_density_history[t]

            if 'vehicle_density_grid' in data and 'uav_density_grid' in data:
                vehicle_density_grid = np.array(data['vehicle_density_grid'])
                uav_density_grid = np.array(data['uav_density_grid'])

                extent = [0, vehicle_density_grid.shape[1], 0, vehicle_density_grid.shape[0]]

                # 绘制车辆密度
                ax_vehicle = axes[i, 0]
                im_vehicle = ax_vehicle.imshow(vehicle_density_grid, extent=extent, origin='lower', cmap='Blues', alpha=0.8)
                ax_vehicle.set_title(f'Time: {t} - Vehicle Density', fontsize=12)
                ax_vehicle.set_xlabel('X coordinate')
                ax_vehicle.set_ylabel('Y coordinate')
                plt.colorbar(im_vehicle, ax=ax_vehicle, orientation='vertical', label='Vehicle Density')

                # 绘制无人机密度
                ax_uav = axes[i, 1]
                im_uav = ax_uav.imshow(uav_density_grid, extent=extent, origin='lower', cmap='Greens', alpha=0.8)
                ax_uav.set_title(f'Time: {t} - UAV Density', fontsize=12)
                ax_uav.set_xlabel('X coordinate')
                ax_uav.set_ylabel('Y coordinate')
                plt.colorbar(im_uav, ax=ax_uav, orientation='vertical', label='UAV Density')

        plt.tight_layout()
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"交通密度历史可视化结果已保存到 {output_path}")
        else:
            plt.show()
