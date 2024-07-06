import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, recall_score
from pathlib import Path
import tensorflow as tf
import numpy as np
import optuna
import glob
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Global variables
train_file = Path.home() / "Documents" / "TFG" / "TrainingAndValidation" / "supervised_train.csv"
test_file = Path.home() / "Documents" / "TFG" / "Testing" / "testing.csv"
directory_path = Path.home() / "Documents" / "TFG" / "Testing"
discriminator_save_path = "GAN_models/optimized_discriminator.h5"
generator_save_path = "GAN_models/optimized_generator.h5"

# [1] Data preparation
def load_and_prepare_data(file):
    data = pd.read_csv(file)
    data = data.sample(frac=0.5, random_state=1)
    X = data.drop(columns=['Label']).values
    y = data['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y

# [2] Load all data
def load_all_data():
    X, y = load_and_prepare_data(train_file)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    X_test, y_test = load_and_prepare_data(test_file)

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = load_all_data()

def build_generator(trial):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(100,)))
    model.add(tf.keras.layers.Dense(trial.suggest_categorical('gen_units_1', [128, 256]), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(trial.suggest_float('gen_dropout_1', 0.2, 0.5)))
    model.add(tf.keras.layers.Dense(trial.suggest_categorical('gen_units_2', [128, 256]), activation='relu'))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Dropout(trial.suggest_float('gen_dropout_2', 0.2, 0.5)))
    model.add(tf.keras.layers.Dense(X_train.shape[1], activation='tanh'))
    return model

def build_discriminator(trial):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
    model.add(tf.keras.layers.Dense(trial.suggest_categorical('disc_units_1', [128, 256]), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float('disc_dropout_1', 0.2, 0.5)))
    model.add(tf.keras.layers.Dense(trial.suggest_categorical('disc_units_2', [128, 256]), activation='relu'))
    model.add(tf.keras.layers.Dropout(trial.suggest_float('disc_dropout_2', 0.2, 0.5)))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='binary_crossentropy', metrics=['accuracy'])
    return model

history = {'accuracy': [], 'loss': []}

def objective(trial):
    X_train, X_val, _, y_train, y_val, _ = load_all_data()
    generator = build_generator(trial)
    discriminator = build_discriminator(trial)

    gan = tf.keras.Sequential([generator, discriminator])
    gan.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy')

    epochs = trial.suggest_int('epochs', 1, 2)
    batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])
    half_batch = batch_size // 2

    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    for epoch in range(epochs):
        for _ in range(X_train.shape[0] // batch_size):
            idx = np.random.randint(0, X_train.shape[0], half_batch)
            real_samples = X_train[idx]

            noise = np.random.normal(0, 1, (half_batch, 100))
            gen_samples = generator.predict(noise)

            d_loss_real = discriminator.train_on_batch(real_samples, np.ones((half_batch, 1)))
            d_loss_fake = discriminator.train_on_batch(gen_samples, np.zeros((half_batch, 1)))
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            noise = np.random.normal(0, 1, (batch_size, 100))
            g_loss = gan.train_on_batch(noise, np.ones((batch_size, 1)))

        d_loss_real, d_acc_real = discriminator.evaluate(X_train, y_train, verbose=1)
        val_loss, val_acc = discriminator.evaluate(X_val, y_val, verbose=1)

        history['accuracy'].append(d_acc_real)
        history['loss'].append(d_loss_real)

        if early_stopping.stopped_epoch > 0:
            break

    noise = np.random.normal(0, 1, (100, 100))
    gen_samples = generator.predict(noise)

    d_loss_real, d_acc_real = discriminator.evaluate(X_val, y_val, verbose=1)
    d_loss_fake, d_acc_fake = discriminator.evaluate(gen_samples, np.zeros((100, 1)), verbose=1)

    y_pred = (discriminator.predict(X_test) > 0.5).astype("int32")
    detection_rate = recall_score(y_test, y_pred)

    trial.set_user_attr("discriminator", discriminator.to_json())
    trial.set_user_attr("generator", generator.to_json())

    return detection_rate

def evaluate_detection_rate(discriminator, directory_path):
    csv_files = glob.glob(str(directory_path / "*.csv"))
    detection_rates = []

    for file in csv_files:
        data = pd.read_csv(file)
        X = data.drop(columns=['Label']).values
        y = data['Label'].values

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        y_pred = (discriminator.predict(X) > 0.5).astype("int32")
        test_loss, test_accuracy = discriminator.evaluate(X, y)
        print(test_accuracy, test_loss)
        print(classification_report(y, y_pred, digits=4))
        detection_rate = recall_score(y, y_pred)
        detection_rates.append(detection_rate)
        print(f'{file}: Detection Rate = {detection_rate}')

    avg_detection_rate = np.mean(detection_rates)
    print(f'Average Detection Rate: {avg_detection_rate}')

def plot():
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 1, 1)
    plt.plot(history['accuracy'], label='Discriminator Train Accuracy')
    plt.title('Discriminator Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    plt.subplot(2, 1, 2)
    plt.plot(history['loss'], label='Discriminator Train Loss')
    plt.title('Discriminator Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=1)

    best_params = study.best_trial.params
    print('Number of finished trials:', len(study.trials))
    print('Best trial:', study.best_trial.params)

    best_discriminator_json = study.best_trial.user_attrs["discriminator"]
    best_generator_json = study.best_trial.user_attrs["generator"]

    best_discriminator = tf.keras.models.model_from_json(best_discriminator_json)
    best_generator = tf.keras.models.model_from_json(best_generator_json)

    best_discriminator.compile(optimizer=tf.keras.optimizers.Adam(), loss='binary_crossentropy', metrics=['accuracy'])
    best_generator.compile(optimizer=tf.keras.optimizers.Adam())

    best_discriminator.save(discriminator_save_path)
    best_generator.save(generator_save_path)

    test_loss, test_accuracy = best_discriminator.evaluate(X_test, y_test, verbose=1)
    print(f'Test Accuracy: {test_accuracy}')

    y_pred = (best_discriminator.predict(X_test) > 0.5).astype("int32")
    print(classification_report(y_test, y_pred))

    evaluate_detection_rate(best_discriminator, directory_path)
    plot()