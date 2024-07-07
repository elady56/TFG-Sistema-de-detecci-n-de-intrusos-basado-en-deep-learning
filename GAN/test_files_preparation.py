import pandas as pd
from pathlib import Path
import glob
import os

# Global variables
directory_path = Path.home() / "Documents" / "TFG" / "Test2"
testing_file = Path.home() / "Documents" / "TFG" / "Testing"
save_path = Path.home() / "Documents" / "TFG" / "Test"
csv_files = glob.glob(str(directory_path / "*.csv"))


benign_data = pd.read_csv(testing_file/"testing.csv")
filtered_df = benign_data[benign_data["Label"] == 0]


print(benign_data.groupby("Label").size())

for file in csv_files:
    file_name = os.path.basename(file)
    print(file_name)
    df = pd.read_csv(file)
    random_half = filtered_df.sample(n=len(df), random_state=42)
    filtered_df = filtered_df.drop(random_half.index)
    df = pd.concat([df, random_half])
    df =df.sample(frac=1).reset_index(drop=True)
    df.to_csv(save_path/file_name, index = False)
    print(df.groupby("Label").size())
    print(len(filtered_df))