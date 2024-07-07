import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import optuna
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

# Global variable
model_path = 'base_GAN_model/optimized_discriminator1.h5'
train_file = Path.home() / "Documents" / "TFG" / "TrainingAndValidation" / "validation.csv"
test_file = Path.home() / "Documents" / "TFG" / "Testing" / "testing.csv"



def load_and_prepare_data(file):
    data = pd.read_csv(file)
    data = data.sample(frac=1, random_state=1)
    X = data.drop(columns=['Label']).values
    y = data['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

#Global variable
loaded_model = load_model(model_path)
X, y = load_and_prepare_data(train_file)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
X_test, y_test = load_and_prepare_data(test_file)
def build_model(trial):
    new_model = Sequential()

    for layer in loaded_model.layers:
        new_model.add(layer)

    new_model.add(Dense(trial.suggest_int('units', 32, 128), activation='relu', name='new_dense3'))
    new_model.add(Dropout(trial.suggest_categorical('dropout', [0.2, 0.3, 0.4, 0.5]), name='new_dropout3'))
    new_model.add(Dense(1, activation='sigmoid', name='new_output3'))


    learning_rate = trial.suggest_categorical('learning_rate', [1e-5, 1e-4,1e-3, 1e-2])
    new_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                      loss='binary_crossentropy', metrics=['accuracy'])

    return new_model


def objective(trial):
    model = build_model(trial)
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    epochs = trial.suggest_int('epochs', 10, 25)
    batch_size = trial.suggest_categorical('batch_size', [32,64,128, 256])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val),
                        callbacks=[early_stopping], verbose=1)

    val_accuracy = history.history['val_accuracy'][-1]
    return val_accuracy

def plot_histories(histories):
    # Accuracy
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['accuracy'], label=f'Model {i+1} Train Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('GAN Model Accuracy Over Epochs')
    plt.show()

    # Loss
    plt.figure(figsize=(12, 6))
    for i, history in enumerate(histories):
        plt.plot(history.history['loss'], label=f'Model {i+1} Train Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('GAN Model Loss Over Epochs')
    plt.show()


if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=3)

    # 3 Best models
    top_trials = study.trials_dataframe().sort_values(by='value', ascending=False).head(3)
    top_models = []
    histories = []

    for i, trial in enumerate(top_trials.itertuples()):
        params = {
            'units': trial.params_units,
            'dropout': trial.params_dropout,
            'learning_rate': trial.params_learning_rate,
            'epochs': trial.params_epochs,
            'batch_size': trial.params_batch_size
        }

        model = build_model(optuna.trial.FixedTrial(params))

        early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

        history = model.fit(X_train, y_train, epochs=params['epochs'], batch_size=params['batch_size'], validation_data=(X_val, y_val),
                            callbacks=[early_stopping])
        # Save the model
        model_path = f'Final_GAN_models/GAN_model_{i + 1}.h5'
        model.save(model_path)
        top_models.append((model, params))
        histories.append(history)

        test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
        print(f'Model {i + 1} Test Accuracy: {test_accuracy}')

        y_pred = (model.predict(X_test) > 0.5).astype("int32")
        print(f'Model {i + 1} Classification Report:')
        print(classification_report(y_test, y_pred, digits=4))

        # Plot
    plot_histories(histories)