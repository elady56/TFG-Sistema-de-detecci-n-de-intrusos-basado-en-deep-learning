from pathlib import Path
import pandas as pd
from tabulate import tabulate
import random


input_path = Path.home() / "TFG" / "Dataset_OK" / "CSV"
output_path = Path.home() / "TFG" / "TFG-Sistema-de-detecci-n-de-intrusos-basado-en-deep-learning"/"BasicAlgorithm"/"Results"

def create_table(input_file, output_file):
    #Real value
    df = pd.read_csv(input_file)
    propotion = df[' Label'].value_counts(normalize=True)
    propotion = propotion.reset_index().transpose()
    propotion.columns = propotion.iloc[0]
    propotion = propotion[1:]
    propotion = propotion.rename(index={'proportion': 'Real'})
    values = df[' Label'].value_counts()
    total=values.sum()

    cnt=1
    for i in values.index:
        propotion.insert(cnt*2-1, i + "_VALUE", values.values[cnt-1])
        cnt+=1

    #Positive
    table= propotion
    positive_row=pd.Series(0, index=table.columns, name='Positive')
    table=table._append(positive_row)
    table.at['Positive','BENIGN']=1.0
    table.at['Positive', 'BENIGN_VALUE'] = total


    #Negative
    negative_row=pd.Series(0, index=table.columns, name='Negative')
    table = table._append(negative_row)
    table.at['Negative', table.columns[0]] = 1.0
    table.at['Negative', table.columns[1]] = total

    #Random
    acum=0
    for i in range(len(values.index)):
        tmp = total - acum if i == len(values) - 1 else random.randint(0, total - acum)
        table.at['Random',values.index[i]+"_VALUE"]=tmp
        table.at['Random',values.index[i]]=tmp/total
        acum+=tmp

    #save
    print(output_path/output_file)
    table.to_csv(output_path/output_file, index=True)

    #print(table)

    tabla = tabulate(table, headers='keys', tablefmt='grid', showindex=True)
    print(input_file)
    print(tabla)
    print("-------------------------")
    print()


if __name__ == '__main__':
    for dir in  Path(input_path).iterdir():
        for file in Path(dir).rglob('*.csv'):
            input_file = file
            output_file = dir.name+"/"+file.name
            create_table(input_file, output_file)




