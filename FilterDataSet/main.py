from pathlib import Path
import pandas as pd

#Variables
original_dir = Path.home() / "TFG" / "Dataset" / "CSV"
save_dir=Path.home() / "TFG" / "Dataset_OK" / "CSV"
features_dir=Path.home() / "TFG"/ "Features.txt"

def process_csv_file(dir_name,file, desired_cols):
    df = pd.read_csv(file)
    print(df.columns)

    df = df[desired_cols]
    save_path = save_dir/dir_name/file.name
    #print(save_path)
    df.to_csv(save_path, index=False)

def process_csvs_in_directory(directory, desired_cols):
    for dir in  Path(directory).iterdir():
        for file in Path(dir).rglob('*.csv'):
            process_csv_file(dir.name,file, desired_cols)

if __name__ == "__main__":
    #Read desired columns
    with open(features_dir, 'r') as file:
        desired_cols= [line.strip("\n") for line in file.readlines() if line.strip()]
        #print(desired_cols)

    process_csvs_in_directory(original_dir, desired_cols)



