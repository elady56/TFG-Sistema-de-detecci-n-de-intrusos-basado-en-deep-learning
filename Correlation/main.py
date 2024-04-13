import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

path = Path.home()/"TFG"/"Dataset_OK"/"CSV"

if __name__ == '__main__':
    data = pd.read_csv(path/"CSV-01-12"/"DrDoS_DNS.csv")
    data = pd.get_dummies(data)
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation matrix')
    plt.show()

