import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

initial_second_day_test_path = Path.home() / "TFG" / "Dataset" / "CSV-03-11"
final_first_day_train_path =  Path.home() / "TFG" / "Dataset" / "TrainingAndValidation"
second_day_test_path = Path.home() / "TFG" / "Dataset" / "Testing"
#[1] Downsample
def downsample_first(number):
    BENIGN_data = pd.DataFrame()
    total = pd.DataFrame()

    for file in Path(initial_second_day_test_path).rglob('*.csv'):

        print(file.name)
        df = pd.read_csv(file)
        df = df.rename(columns=lambda x: x.strip() if isinstance(x, str) and x.strip() else x)
        print(df.groupby("Label").size())
        BENIGN_data = pd.concat([df[df['Label'] == 'BENIGN'], BENIGN_data], ignore_index=True)

        if file.name == "Syn.csv":
            Syn_data=(df[df['Label'] == 'Syn']).sample(n=number, random_state=42)
            Syn_data = initial_treatment(Syn_data)
            Syn_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, Syn_data], ignore_index=True)
            print(total.groupby("Label").size())
        elif file.name == "UDP.csv":
            UDP_data=(df[df['Label'] == 'UDP']).sample(n=number, random_state=42)
            UDP_data = initial_treatment(UDP_data)
            UDP_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, UDP_data], ignore_index=True)
        elif file.name == "UDPLag.csv":
            UDPLag_data = (df[df['Label'] == 'UDPLag']) # Only 1873
            UDPLag_data = initial_treatment(UDPLag_data)
            UDPLag_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, UDPLag_data], ignore_index=True)
        elif file.name == "MSSQL.csv":
            MSSQL_data = (df[df['Label'] == 'MSSQL']).sample(n=number, random_state=42)
            MSSQL_data = initial_treatment(MSSQL_data)
            MSSQL_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, MSSQL_data], ignore_index=True)
        elif file.name == "NetBIOS.csv":
            NetBIOS_data = (df[df['Label'] == 'NetBIOS']).sample(n=number, random_state=42)
            NetBIOS_data = initial_treatment(NetBIOS_data)
            NetBIOS_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, NetBIOS_data], ignore_index=True)
        elif file.name == "LDAP.csv":
            LDAP_data = (df[df['Label'] == 'LDAP']).sample(n=number, random_state=42)
            LDAP_data = initial_treatment(LDAP_data)
            LDAP_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, LDAP_data], ignore_index=True)
        elif file.name == "Portmap.csv":
            Portmap_data = (df[df['Label'] == 'Portmap']).sample(n=9022, random_state=42)
            Portmap_data = initial_treatment(Portmap_data)
            Portmap_data.to_csv(second_day_test_path / file.name, index=False)
            total = pd.concat([total, Portmap_data], ignore_index=True)

    BENIGN_data = BENIGN_data.sample(n=56000, random_state=42)
    BENIGN_data = initial_treatment(BENIGN_data)
    total = pd.concat([total, BENIGN_data], ignore_index=True)

    print(total.groupby("Label").size())
    total.to_csv(second_day_test_path / "testing.csv", index=None)

#[2] Initial treatment
def initial_treatment(df):

    # [2.1] Initial Delete
    desired_columns = pd.read_csv(final_first_day_train_path/"features.txt", header=None).iloc[:, 0].tolist()
    df = df[desired_columns]

    #[2.2] String treatment - IP
    IPs = set()
    IPs.update(df['Source IP'])
    IPs.update(df['Destination IP'])
    ip_index_map = {ip: index for index, ip in enumerate(IPs)}
    for column in ['Source IP', 'Destination IP']:
        df[column] = df[column].map(ip_index_map)

    #[2.3] Null value treatment - replace them with 0
    null_columns = df.isnull().sum()
    null_columns = null_columns[null_columns > 0]
    df[null_columns.index] = df[null_columns.index].fillna(0)
    #print(null_columns)


    #[2.4] Inf value treatment - replace them with -1

    #numeric_columns = df.select_dtypes(include=[np.number]).columns
    #inf_columns = np.isinf(df[numeric_columns]).sum()
    #inf_columns = inf_columns[inf_columns > 0]
    #print(inf_columns)

    df.replace([np.inf, -np.inf], -1, inplace=True)

    # [2.5] Replace Label with 0/1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    return df


if __name__ == "__main__":
    #[1] Downsample
    downsample_first(9021)









