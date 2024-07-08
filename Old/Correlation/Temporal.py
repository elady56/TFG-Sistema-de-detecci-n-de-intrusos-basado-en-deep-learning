import pandas as pd
from pathlib import Path

#path = Path.home()/"TFG"/"Dataset_OK"/"CSV"/"Total-First-Day"
path = Path.home()/"TFG"/"Dataset_OK"/"CSV"/"Read"/"CSV-01-12"



if __name__ == "__main__":
    df = pd.read_csv(path / "total_final.csv")
    print(df.groupby('Label').size())

    #df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    #df.to_csv(path /"total_to_train.csv", index=None)

    df = pd.read_csv(path / "total_to_train.csv")
    print(df.groupby('Label').size())


