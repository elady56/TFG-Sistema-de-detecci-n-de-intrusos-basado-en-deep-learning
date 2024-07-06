from pathlib import Path
import pandas as pd
import random


input_path = Path.home() / "TFG" / "Dataset" / "Testing"
output_path = Path.home() / "TFG" / "TFG-Sistema-de-detecci-n-de-intrusos-basado-en-deep-learning"/"BasicAlgorithm"/"Results"

"""def create_table(input_file, output_file):
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
    print()"""

def random_algorithm(file):
    df = pd.read_csv(file)
    labels = df['Label']
    total_benign = 0.0
    total_attack = 0.0
    detected_benign = 0.0
    detected_attack = 0.0

    for l in labels:
        aux= random.randint(0,1)
        if l == 0:
            total_benign+=1
            if aux == 0:
                detected_benign+=1
        else:
            total_attack+=1
            if aux == 1:
                detected_attack+=1
    return total_benign,total_attack,detected_benign,detected_attack


def always_true(file):
    df = pd.read_csv(file)
    labels = df['Label']
    total_benign = 0.0
    total_attack = 0.0
    detected_benign = 0.0
    detected_attack = 0.0

    for l in labels:
        aux= 1
        if l == 0:
            total_benign+=1
            if aux == 0:
                detected_benign+=1
        else:
            total_attack+=1
            if aux == 1:
                detected_attack+=1
    return total_benign,total_attack,detected_benign,detected_attack


def always_false(file):
    df = pd.read_csv(file)
    labels = df['Label']
    total_benign = 0.0
    total_attack = 0.0
    detected_benign = 0.0
    detected_attack = 0.0

    for l in labels:
        aux= 0
        if l == 0:
            total_benign+=1
            if aux == 0:
                detected_benign+=1
        else:
            total_attack+=1
            if aux == 1:
                detected_attack+=1
    return total_benign,total_attack,detected_benign,detected_attack


if __name__ == '__main__':
    output = open("results.txt", "w")
    #CSV-01-12
    total_benign = 0.0
    total_attack = 0.0
    detected_benign = 0.0
    detected_attack = 0.0

    for file in Path(input_path).rglob('*.csv'):
        values=always_true(file)
        total_benign+=values[0]
        total_attack+=values[1]
        detected_benign+=values[2]
        detected_attack+=values[3]
        print(str(round(detected_benign/total_benign,4)))

    output.write("Normal: " + str(round(detected_benign/total_benign,4))+"\n")
    output.write("Attack: " + str(round(detected_attack/ total_attack,4)) + "\n")

    #CSV-03-11
    for file in Path(input_path/"Attacks").rglob('*.csv'):
        print(file.name)
        values = always_true(file)
        total_benign = values[0]
        total_attack = values[1]
        detected_benign = values[2]
        detected_attack = values[3]
        output.write(file.name.replace(".csv","")+": " + str(round(detected_attack / total_attack, 4)) + "\n")

    output.close()

    """for dir in  Path(input_path).iterdir():
    for file in Path(dir).rglob('*.csv'):
    input_file = file
    output_file = dir.name+"/"+file.name
    create_table(input_file, output_file)"""



