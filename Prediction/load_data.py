import pandas as pd
import json
import os

def load_data(data_path: str, context_path: str) -> tuple[list[list[str]]]:
    if not data_path.endswith('.csv'):
        raise ValueError(f"Invalid data file: {data_path}. Expected a CSV file.")
    if not context_path.endswith('.json'):
        raise ValueError(f"Invalid context file: {context_path}. Expected a JSON file.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if not os.path.exists(context_path):
        raise FileNotFoundError(f"Context file not found: {context_path}")

    df = pd.read_csv(data_path)
    data = df.to_dict(orient='list')

    all_values = list(data.values())

    with open(context_path, 'r') as f:
        context_summary = json.load(f)

    y1 = all_values[7][1:]  # compute_load_avg
    x11 = all_values[1][:len(y1)]   # current_task_success_ratio
    x12 = all_values[8][:len(y1)]   # current_vehicle_density
    x13 = all_values[9][:len(y1)]   # current_uav_density
    x14 = all_values[10][:len(y1)]  # junction_0_vehicle_count
    x15 = all_values[11][:len(y1)]  # junction_1_vehicle_count
    x16 = all_values[12][:len(y1)]  # junction_2_vehicle_count

    y2 = all_values[4][1:]  # avg_V2U_rate
    x21 = all_values[8][:len(y2)]   # current_vehicle_density
    x22 = all_values[9][:len(y2)]   # current_uav_density
    x23 = all_values[7][:len(y2)]  # current_compute_load_avg

    y3 = all_values[1][1:]  # task_success_ratio
    x31 = all_values[4][:len(y3)]   # current_avg_V2U_rate
    x32 = all_values[5][:len(y3)]   # current_avg_V2I_rate
    x33 = all_values[7][:len(y3)]   # current_compute_load_avg

    y4 = all_values[9][1:]  # uav_density
    x41 = all_values[6][:len(y4)]   # current_avg_U2I_rate
    x42 = all_values[7][:len(y4)]  # current_compute_load_avg
    x43 = [context_summary["nonfly_zones"]["area_ratio"]] * len(y4)    # nonfly_zones.area_ratio

    dep_vars = list(map(str, [y1, y2, y3, y4]))
    indep_vars = list(map(list, (map(str, [x11, x12, x13, x14, x15, x16]),
                                    map(str, [x21, x22, x23]),
                                    map(str, [x31, x32, x33]),
                                    map(str, [x41, x42, x43]))))
    
    return indep_vars, dep_vars
