import pandas as pd
import os

def load_data(data_path: str) -> tuple[list[list[str]]]:
    if not data_path.endswith('.csv'):
        raise ValueError(f"Invalid data file: {data_path}. Expected a CSV file.")
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Data file not found: {data_path}")

    df = pd.read_csv(data_path)
    data = df.to_dict(orient='list')

    all_values = list(data.values())

    y1 = all_values[7][1:]  # compute_load_avg
    x11 = all_values[1][:len(y1)]   # current_task_success_ratio
    x12 = all_values[8][:len(y1)]   # current_vehicle_density
    x13 = all_values[4][:len(y1)]   # current_avg_V2U_rate
    x14 = all_values[7][:len(y1)]   # current_compute_load_avg
    x15 = [0] + all_values[7][:len(y1) - 1]  # previous_compute_load_avg

    y2 = all_values[4][1:]  # avg_V2U_rate
    x21 = all_values[8][:len(y2)]   # current_vehicle_density
    x22 = all_values[9][:len(y2)]   # current_uav_density
    x23 = all_values[7][:len(y2)]  # current_compute_load_avg

    y3 = all_values[1][1:]  # task_success_ratio
    x31 = all_values[4][:len(y3)]   # current_avg_V2U_rate
    x32 = all_values[5][:len(y3)]   # current_avg_V2I_rate
    x33 = all_values[7][:len(y3)]   # current_compute_load_avg

    dep_vars = list(map(str, [y1, y2, y3]))
    indep_vars = list(map(list, (map(str, [x11, x12, x13, x14, x15]),
                                    map(str, [x21, x22, x23]),
                                    map(str, [x31, x32, x33]))))
    
    return indep_vars, dep_vars
