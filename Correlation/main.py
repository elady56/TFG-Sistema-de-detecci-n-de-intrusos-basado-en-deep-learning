import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

path = Path.home()/"TFG"/"Dataset_OK"/"CSV"
path_output = Path.home()/"TFG"/"TFG-Sistema-de-detecci-n-de-intrusos-basado-en-deep-learning"/"Correlation"/"output"
def joinCSV():
    df = pd.DataFrame()
    for file in Path(path/"CSV-01-12").rglob('*.csv'):
        tmp = pd.read_csv(file)
        print(file.name)
        print(tmp.shape)
        tmp["type"]=file.name.replace(".csv","")
        df = pd.concat([df, tmp], ignore_index=True)
    df.to_csv(path / "Total-First-Day"/ "total.csv", index=None)

def transformNoNumericalColumnsAndNullValue():
    df = pd.read_csv(path / "Total-First-Day" / "total.csv")
    IPs = set()
    IPs.update(df['Source IP'])
    IPs.update(df['Destination IP'])
    ip_index_map = {ip: index for index, ip in enumerate(IPs)}

    for column in ['Source IP', 'Destination IP']:
        df[column] = df[column].map(ip_index_map)
    df['Flow Bytes/s'] = df ['Flow Bytes/s'].fillna(0)
    df.to_csv(path / "Total-First-Day" / "total2.csv", index=None)

def dropColumns():
    df = pd.read_csv(path_output / "columns.csv")
    df = df.reindex(df['value'].abs().sort_values(ascending=False).index)
    print(df)
    remove=set()
    keep=set()
    for row in df.iterrows():
        if row[1]["param1"] not in keep:
            remove.add(row[1]["param1"])
            keep.add(row[1]["param2"])
        elif row[1]["param2"] not in keep:
            remove.add(row[1]["param2"])

    df = pd.read_csv(path / "Total-First-Day"  / "total2.csv")
    df = df.drop (columns=list(remove), axis=1)
    df.to_csv(path / "total_final.csv", index=None)
    correlation_matrix = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation End matrix')
    plt.show()

def columnFilter():
    df = pd.read_csv(path_output / "correlation.csv")
    threshold = 0.5
    re = pd.DataFrame(columns=["param1", "param2", "value"])
    for row in df.index:
        for column in df.columns[row + 2:]:
            if abs(df.loc[row, column]) >= threshold:
                re.loc[len(re)] = [df.loc[row][0], column, df.loc[row, column]]
    re.to_csv(path_output / "columns.csv", index=None)

def correlationIni():
    df = pd.read_csv(path / "Total-First-Day" / "total2.csv")
    print(df.shape)
    correlation_matrix = df.corr()
    correlation_matrix.to_csv("output/correlation.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation Ini matrix')
    plt.show()


if __name__ == '__main__':
    #joinCSV()
    #transformNoNumericalColumnsAndNullValue()
    #correlationIni()
    #columnFilter()
    #dropColumns()
    df = pd.read_csv(path / "Total-First-Day" / "total2.csv")
    print(df.groupby("Bwd PSH Flags").size())




