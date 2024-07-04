from pathlib import Path
import pandas as pd


path=Path.home() / "TFG" / "Dataset_OK" / "CSV" /"Read"

if __name__ == "__main__":
    for dir in Path(path).iterdir():
        for file in Path(dir).rglob('*.csv'):
            print(file.name)
            df = pd.read_csv(file)
            print(df.groupby("Label").size())


