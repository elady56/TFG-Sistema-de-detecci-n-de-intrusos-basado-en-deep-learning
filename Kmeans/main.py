from pathlib import Path
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import csv
path = Path.home()/"TFG"/"Dataset_OK"/"CSV"

#def read_csv_files():




if __name__ == "__main__":
    """for dir in Path(path).iterdir():
        for file in Path(dir).rglob('*.csv'):
            home_data = pd.read_csv(file)
            home_data.head()
            print(home_data.columns)"""

    home_data = pd.read_csv(path/"CSV-01-12"/"DrDoS_DNS.csv")
    sns.scatterplot(data=home_data, x='Total Length of Fwd Packets', y='Fwd Packets/s', hue=' Label')

    plt.show()

