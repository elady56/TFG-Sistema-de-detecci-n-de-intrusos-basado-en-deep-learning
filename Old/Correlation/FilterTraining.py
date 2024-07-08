from pathlib import Path
import pandas as pd


original_dir = Path.home() / "TFG" / "Dataset_OK" / "CSV" /"CSV-03-11"
save_dir=Path.home() / "TFG" / "Dataset_OK" / "CSV"/ "Second-Day-Test"
features_dir=Path.home() / "TFG"/ "final_features.txt"

def transformNoNumericalColumnsAndNullValue(df):
    IPs = set()
    IPs.update(df['Source IP'])
    IPs.update(df['Destination IP'])
    ip_index_map = {ip: index for index, ip in enumerate(IPs)}
    for column in ['Source IP', 'Destination IP']:
        df[column] = df[column].map(ip_index_map)
    df['Flow Bytes/s'] = df ['Flow Bytes/s'].fillna(0)
    return df

def process_csv_file(dir_name,file, desired_cols):
    df = pd.read_csv(file)
    df = transformNoNumericalColumnsAndNullValue(df)
    df = df[desired_cols]
    df = df.rename(columns=lambda x: x.strip())
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
    save_path = save_dir/dir_name/file.name
    print(save_path)
    df.to_csv(save_path, index=False)

def process_csvs_in_directory(directory, desired_cols):
    for file in Path(original_dir).rglob('*.csv'):
        process_csv_file(original_dir.name, file, desired_cols)

if __name__ == "__main__":
    #Read desired columns

    with open(features_dir, 'r') as file:
        desired_cols= [line.strip("\n") for line in file.readlines() if line.strip()]
        print(desired_cols)

    process_csvs_in_directory(original_dir, desired_cols)