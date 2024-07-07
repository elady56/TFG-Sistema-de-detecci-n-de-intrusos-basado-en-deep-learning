import pandas as pd
import numpy as np
import glob
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, recall_score, precision_score, f1_score, roc_curve, auc

from pathlib import Path
from sklearn.preprocessing import StandardScaler

test_directory = Path.home() / "Documents" / "TFG" / "Test"
gan_model_path = Path.cwd()/"Final_GAN_models"
lstm_model_path = Path.cwd()/"Final_LSTM_models"
output_path = Path.cwd()/"Results"
def load_and_prepare_data(file):
    data = pd.read_csv(file)
    data = data.sample(frac=1, random_state=42)
    X = data.drop(columns=['Label']).values
    y = data['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

def evaluate_model_on_multiple_files(output_file, model, test_directory, is_lstm):
    all_y_true = []
    all_y_pred = []
    detection_rates = []

    test_files = glob.glob(str(test_directory / "*.csv"))
    with open(output_file, 'w') as f:
        for file in test_files:
            X, y = load_and_prepare_data(file)
            if is_lstm:
                X = X.reshape((X.shape[0], 1, X.shape[1]))

            y_pred = (model.predict(X) > 0.5).astype("int32")
            all_y_true.extend(y)
            all_y_pred.extend(y_pred)

            test_loss, test_accuracy = model.evaluate(X, y, verbose=0)
            f.write(f'File: {file} - Test Accuracy: {test_accuracy}, Test Loss: {test_loss}\n')
            report = classification_report(y, y_pred, digits=4)
            f.write(report + '\n')
            detection_rate = recall_score(y, y_pred)
            detection_rates.append(detection_rate)
            f.write(f'{file}: Detection Rate = {detection_rate}\n')

        accuracy = accuracy_score(all_y_true, all_y_pred)
        recall = recall_score(all_y_true, all_y_pred)
        precision = precision_score(all_y_true, all_y_pred)
        f1 = f1_score(all_y_true, all_y_pred)
        avg_detection_rate = np.mean(detection_rates)

        f.write(f'Overall Accuracy: {accuracy}\n')
        f.write(f'Overall Recall: {recall}\n')
        f.write(f'Overall Precision: {precision}\n')
        f.write(f'Overall F1 Score: {f1}\n')
        f.write(f'Average Detection Rate: {avg_detection_rate}\n')
        f.write('Overall Classification Report:\n')
        f.write(classification_report(all_y_true, all_y_pred, digits=4))

    return all_y_true,all_y_pred

def plot_ROC_curve(total):
    for key, values in total.items():
        fpr, tpr, _ = roc_curve(values[1], values[2])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, color=values[0], lw=2, label= f'{key} ROC curve (area = %0.2f)' % roc_auc)

    plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC - Comparación de Modelos')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    total = {"TRUE":['skyblue', [1,0],[1,1]], "FALSE":['red',[1,0],[0,0]], "RANDOM":['m',[1,0,0,1],[0,1,0,1]],
            "GAN_model_1": ['darkorange'] , "GAN_model_2": ['blue'], "GAN_model_3": ['yellow'], "LSTM_model_1": ['green'],
             "LSTM_model_2": ['navy'], "LSTM_model_3": ['purple']}
    #GAN models
    for models in glob.glob(str(gan_model_path/"*.h5")):
        output_file = output_path / (Path(models).stem + ".txt")
        print(output_file)
        gan_model = load_model(models)
        #gan_model.summary()
        all_y_true, all_y_pred = evaluate_model_on_multiple_files(output_file, gan_model, test_directory,0)
        if total.keys().__contains__(Path(models).stem):
            total[Path(models).stem].append(all_y_true)
            total[Path(models).stem].append(all_y_pred)

    #LSTM models
    for models in glob.glob(str(lstm_model_path/"*.h5")):
        output_file = output_path / (Path(models).stem + ".txt")
        print(output_file)
        lstm_model = load_model(models)
        all_y_true, all_y_pred = evaluate_model_on_multiple_files(output_file, lstm_model, test_directory,1)

        if total.keys().__contains__(Path(models).stem):
            total[Path(models).stem].append(all_y_true)
            total[Path(models).stem].append(all_y_pred)


    plot_ROC_curve(total)


