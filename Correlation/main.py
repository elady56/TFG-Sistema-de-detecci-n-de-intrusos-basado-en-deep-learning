import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

path = Path.home()/"TFG"/"Dataset_OK"/"CSV"

if __name__ == '__main__':
    df = pd.read_csv(path / "CSV-01-12" / "total.csv")
    print(df.describe().T)
    print(df.shape)
    print(df.groupby("Label").size())
    print(df.isnull().sum())
    correlation_matrix = df.corr()
    correlation_matrix.to_csv("output/correlation.csv")
    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", annot_kws={"size": 10})
    plt.title('Correlation matrix')
    plt.show()

