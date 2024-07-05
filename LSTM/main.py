import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report, recall_score
from pathlib import Path
import tensorflow as tf
import optuna
import glob
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Global variables
train_file = Path.home() / "Documents" / "TFG" / "TrainingAndValidation" / "supervised_train.csv"
test_file = Path.home() / "Documents" / "TFG" / "Testing" / "testing.csv"
directory_path = Path.home() / "Documents" / "TFG" / "Testing"

# [1] Data preparation
def load_and_prepare_data(file):
    data = pd.read_csv(file)
    data = data.sample(frac=1).reset_index(drop=True)

    X = data.drop(columns=['Label']).values
    y = data['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# [2] Load all data
def load_all_data():
    #X_train, y_train = load_and_prepare_data(train_file)
    #X_val, y_val = load_and_prepare_data(val_file)
    X, y = load_and_prepare_data(train_file)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, y_test = load_and_prepare_data(test_file)

    # Reshape for LSTM input
    X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
    X_val = X_val.reshape((X_val.shape[0], 1, X_val.shape[1]))
    X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

    return X_train, X_val, X_test, y_train, y_val, y_test

# [3] Define LSTM model
def build_lstm_model(trial, input_shape):
    model = tf.keras.Sequential()

    n_layers = trial.suggest_int('n_layers', 1, 3)
    for i in range(n_layers):
        units = trial.suggest_int(f'units_lstm_{i}', 50, 200)
        if i == n_layers - 1:
            model.add(tf.keras.layers.LSTM(units, input_shape=input_shape))
        else:
            model.add(tf.keras.layers.LSTM(units, return_sequences=True))

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


    return model

# [4] Objective function for Optuna
def objective(trial):
    X_train, X_val, _, y_train, y_val, _ = load_all_data()
    input_shape = (X_train.shape[1], X_train.shape[2])
    model = build_lstm_model(trial, input_shape)

    batch_size = 64#trial.suggest_int('batch_size', 32, 128)
    epochs = trial.suggest_int('epochs', 2, 5)

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train,  epochs=epochs, batch_size=batch_size, verbose=1,
                        validation_data=(X_val, y_val), callbacks=[early_stopping])


    # Evaluate on validation set to get detection rate
    y_val_pred = (model.predict(X_val) > 0.5).astype("int32")
    detection_rate = recall_score(y_val, y_val_pred)

    return detection_rate

# [5] Train the model with the best parameters
def train_lstm_with_params(params, X_train, X_val, y_train, y_val):
    model = tf.keras.Sequential()
    n_layers = int(params['n_layers'])
    for i in range(n_layers):
        units = int(params[f'units_lstm_{i}'])
        if i == n_layers - 1:
            model.add(tf.keras.layers.LSTM(units, input_shape=(X_train.shape[1], X_train.shape[2])))
        else:
            model.add(tf.keras.layers.LSTM(units, return_sequences=True))

    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    history = model.fit(X_train, y_train, epochs=int(params['epochs']), batch_size=int(params['batch_size']),
                        validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=1)

    return model, history

# [6] Evaluate detection rate on a directory of CSV files
def evaluate_detection_rate(model, directory_path):
    csv_files = glob.glob(str(directory_path / "*.csv"))

    for file in csv_files:
        X, y = load_and_prepare_data(file)
        X = X.reshape((X.shape[0], 1, X.shape[1]))
        y_pred = (model.predict(X) > 0.5).astype("int32")
        test_loss, test_accuracy = model.evaluate(X, y)
        print(f'Model {i} Test Accuracy: {test_accuracy}')
        detection_rate = recall_score(y, y_pred)
        print(f'{file}: Detection Rate = {detection_rate:.4f}')

def plot(histories):
    # Plot accuracy for all models
    plt.figure(figsize=(10, 5))
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Model {i} Train')

    plt.title('Models Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.show()

    # Plot loss for all models
    plt.figure(figsize=(10, 5))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Model {i} Train')

    plt.title('Models Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

if name == "main":
    # Execute the optimization
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=2)

    # Load and prepare data
    X_train, X_val, X_test, y_train, y_val, y_test = load_all_data()

    # Get the top 5 trials
    top_trials = study.trials_dataframe().sort_values(by='value', ascending=False).head(5)
    top_models = []
    histories = []

    for i, trial in enumerate(top_trials.itertuples()):
        params = {
            'n_layers': int(getattr(trial, 'params_n_layers')),
            'epochs': int(getattr(trial, 'params_epochs')),
            'batch_size': int(getattr(trial, 'params_batch_size'))
        }
        for j in range(params['n_layers']):
            params[f'units_lstm_{j}'] = getattr(trial, f'params_units_lstm_{j}')

        model, history = train_lstm_with_params(params, X_train, X_val, y_train, y_val)
        top_models.append((model, params))
        histories.append(history)

        # Save the model
        model.save(f'LSTM_models/optimized_lstm_model_{i}.h5')

        # Evaluate on the test set
        test_loss, test_accuracy = model.evaluate(X_test, y_test)
        print(f'Model {i} Test Accuracy: {test_accuracy}')

        # Print classification report
        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        detection_rate = recall_score(y_test, y_pred)
        print("Detection rate" + str(detection_rate))
        print(f'Model {i} Classification Report:')
        print(classification_report(y_test, y_pred, digits=4))

        print(f'Evaluating detection rate for Model {i}')
        csv_files = glob.glob(str(directory_path / "*.csv"))

        for file in csv_files:
            X, y = load_and_prepare_data(file)
            X = X.reshape((X.shape[0], 1, X.shape[1]))
            y_pred = (model.predict(X) > 0.5).astype("int32")
            print(classification_report(y, y_pred, digits=4))
            detection_rate = recall_score(y, y_pred)
            print(f'{file}: Detection Rate = {detection_rate:.4f}')

    plot(histories)