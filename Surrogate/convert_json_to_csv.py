import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def import_data():
    # Get data
    with open("data_LHS_12.json", "r") as f:
        data = json.load(f)

    # Flatten the dataset: Each entry in rot_angles and torques lists becomes a new row
    flattened_data = []
    for entry in data:
        torques = np.array(entry["torques"])
        avg_T = np.mean(torques)
        ripple_T  = (np.max(torques)-np.min(torques))/avg_T


        flattened_data.append({
            'h_s0': np.round(entry['h_s0'], 4),
            'h_s1': np.round(entry['h_s1'], 4),
            'w_s0': np.round(entry['w_s0'], 4),
            'w_s1': np.round(entry['w_s1'], 4),
            'T_avg': np.round(avg_T,4),
            'T_ripple': np.round(ripple_T, 4),
        })

    # Create DataFrame and save to CSV
    df = pd.DataFrame(flattened_data)
    csv_filename = "dataset.csv"
    df.to_csv(csv_filename, index=False)

    print(f"Dataset successfully converted to {csv_filename}.")
    print(df.head())

    # Output with tabs (\t) as the separator for easy Word pasting
    print(df.to_csv(sep='\t', index=False))

if __name__ == "__main__":
    import_data()