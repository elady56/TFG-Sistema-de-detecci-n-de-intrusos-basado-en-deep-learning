import pandas as pd
from pathlib import Path

path = Path.home()/"TFG"/"Dataset_OK"/"CSV"/"CSV-01-12"

if __name__ == '__main__':
    #df=pd.read_csv(path/"total.csv")
    df = pd.DataFrame()
    #print(df)
    for file in Path(path).rglob('*.csv'):
        tmp = pd.read_csv(file)
        print(file.name)
        print(tmp.shape)
        tmp["type"]=file.name.replace(".csv","")
        df = pd.concat([df, tmp], ignore_index=True)

    print("H")
    df.to_csv(path/"total.csv",index=None)
    #print(df.describe().T)



