import pandas as pd
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

initial_first_day_train_path = Path.home() / "TFG" / "Dataset" / "CSV-01-12"
final_first_day_train_path =  Path.home() / "TFG" / "Dataset" / "TrainingAndValidation"
#[1] Downsample
def downsample_first(number):
    BENIGN_data = pd.DataFrame()
    total = pd.DataFrame()

    for file in Path(initial_first_day_train_path).rglob('*.csv'):
        print(file.name)
        df = pd.read_csv(file,low_memory=False)
        BENIGN_data = pd.concat([df[df[' Label'] == 'BENIGN'], BENIGN_data], ignore_index=True)
        if file.name == "Syn.csv":
            Syn_data=(df[df[' Label'] == 'Syn']).sample(n=number, random_state=42)
            total = pd.concat([total, Syn_data], ignore_index=True)
            print(total.groupby(" Label").size())
        elif file.name == "DrDoS_UDP.csv":
            UDP_data=(df[df[' Label'] == 'DrDoS_UDP']).sample(n=number, random_state=42)
            total = pd.concat([total, UDP_data], ignore_index=True)
        elif file.name == "UDPLag.csv":
            UDPLag_data = (df[df[' Label'] == 'UDP-lag']).sample(n=number, random_state=42)
            total = pd.concat([total, UDPLag_data], ignore_index=True)
        elif file.name == "DrDoS_MSSQL.csv":
            MSSQL_data = (df[df[' Label'] == 'DrDoS_MSSQL']).sample(n=number, random_state=42)
            total = pd.concat([total, MSSQL_data], ignore_index=True)
        elif file.name == "DrDoS_NetBIOS.csv":
            NetBIOS_data = (df[df[' Label'] == 'DrDoS_NetBIOS']).sample(n=number, random_state=42)
            total = pd.concat([total, NetBIOS_data], ignore_index=True)
        elif file.name == "DrDoS_LDAP.csv":
            LDAP_data=(df[df[' Label'] == 'DrDoS_LDAP']).sample(n=number, random_state=42)
            total = pd.concat([total, LDAP_data], ignore_index=True)

    BENIGN_data = BENIGN_data.sample(n=72000, random_state=42)

    total = pd.concat([total, BENIGN_data], ignore_index=True)
    print("Nice")
    total=total.rename(columns=lambda x: x.strip() if isinstance(x, str) and x.strip() else x)

    print(total.groupby("Label").size())
    total.to_csv(initial_first_day_train_path / "total.csv", index=None)

#[2] Initial treatment
def initial_treatment():

    undesired_columns = ["Unnamed: 0", "Flow ID", "Timestamp","SimillarHTTP"]
    """undesired_columns = ["Unnamed: 0", "Flow ID","FIN Flag Count", "Timestamp", "SYN Flag Count", "RST Flag Count", "PSH Flag Count",
                         "ACK Flag Count", "URG Flag Count", "CWE Flag Count", "ECE Flag Count","SimillarHTTP","Inbound",
                         "Idle Min", "act_data_pkt_fwd", "Fwd Header Length.1", "Fwd URG Flags", "Bwd URG Flags",
                         "Init_Win_bytes_backward", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk",
                         "Bwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bulk Rate"]"""


    #[2.1] Initial Delete
    df = pd.read_csv(final_first_day_train_path/"total.csv")
    df = df.drop(columns=undesired_columns, axis=1)

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

    #[2.5] Replace Label with 0/1
    df['Label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)

    return df

#[3] Initial correlation
def correlation(df, fig_name):
    df = df.drop(columns=['Label'])
    print(df.shape)
    correlation_matrix = df.corr()
    correlation_matrix.to_csv("output/correlation_matrix.csv")
    plt.figure(figsize=(18, 20))
    mask = np.triu(np.ones_like(correlation_matrix, dtype=bool))
    sns.heatmap(correlation_matrix, mask=mask, cmap='coolwarm', annot=False)
    plt.title(fig_name)
    plt.savefig(f'output/{fig_name}.pdf', format='pdf')
    plt.show()

#[4] Columns filter
def columnFilter():
    df = pd.read_csv("output/correlation_matrix.csv")
    threshold = 0.7
    re = pd.DataFrame(columns=["param1", "param2", "value"])

    for row in df.index:
        for column in df.columns[row + 2:]:
            if abs(df.loc[row, column]) >= threshold:
                re.loc[len(re)] = [df.loc[row][0], column, df.loc[row, column]]

    re = re.reindex(re['value'].abs().sort_values(ascending=False).index)
    re['value'] = re['value'].apply(lambda x: f"{x:.4f}")
    re.to_csv("output/correlation_pair_above_threshold.csv", index=None)

#[5] Drop high correlated columns and generate training and validation dataset
def dropColumns(df):
    """columns = pd.read_csv("output/correlation_pair_above_threshold.csv")
    remove=set()
    keep=set()

    for row in columns.iterrows():
        if row[1]["param1"] not in keep:
            remove.add(row[1]["param1"])
            keep.add(row[1]["param2"])
        elif row[1]["param2"] not in keep:
            remove.add(row[1]["param2"])"""

    remove = ["FIN Flag Count", "SYN Flag Count", "RST Flag Count", "PSH Flag Count","ACK Flag Count", "URG Flag Count",
              "CWE Flag Count", "ECE Flag Count", "Inbound","Idle Min", "act_data_pkt_fwd", "Fwd Header Length.1", "Fwd URG Flags",
              "Bwd URG Flags","Init_Win_bytes_backward", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Bwd Avg Bytes/Bulk",
              "Bwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bulk Rate", "Subflow Fwd Bytes", "Subflow Bwd Bytes",
              "Subflow Bwd Packets", "Subflow Fwd Packets","RST Flag Count", "Idle Max", "Avg Fwd Segment Size",
              "Avg Bwd Segment Size", "Min Packet Length", "Flow Duration", "Average Packet Size", "Init_Win_bytes_forward",
              "Idle Mean", "Total Backward Packets", "min_seg_size_forward", "Flow IAT Max",
              "Flow IAT Min", "Packet Length Mean", "Packet Length Std", "Flow IAT Std", "Total Fwd Packets",
              "Flow IAT Std", "Bwd IAT Std", "Fwd IAT Std" ]

    df = df.drop (columns=remove, axis=1)
    print(df.shape)
    #print(df.describe().T)
    #print(df.columns)

    BENIGN_data = df[df['Label'] == 0]
    train_data, validation_benign_data = train_test_split(BENIGN_data, test_size=0.2, random_state=42)

    validation_attack_data=df.drop(train_data.index)
    validation_data = pd.concat([validation_attack_data, validation_benign_data], ignore_index=True)

    train_data.to_csv(final_first_day_train_path/ "train.csv", index=None)
    validation_data.to_csv(final_first_day_train_path/ "validation.csv", index=None)

    return df

#[7] Features extraction

def features_extraction():
    df = pd.read_csv(final_first_day_train_path/"train.csv")

    column_names = df.columns.tolist()
    pd.DataFrame(column_names).to_csv(final_first_day_train_path / "features.txt", header=False, index=False)
    print(df.columns)

if __name__ == "__main__":
    #[1] Downsample
    #downsample_first(12000)

    #[2] Initial treatment
    initial_treated_df = initial_treatment()
    print(initial_treated_df.groupby("Label").size())

    #[3] Initial correlation
    correlation(initial_treated_df, "Initial Correlation Matrix")

    #[4] Columns filter
    columnFilter()

    #[5] Drop high correlated columns and generate training and validation dataset
    correlated_df = dropColumns(initial_treated_df)

    #[6] Final Correlation
    correlation(correlated_df, "Final Correlation Matrix")
    columnFilter()

    #[7] Features extraction
    features_extraction()






