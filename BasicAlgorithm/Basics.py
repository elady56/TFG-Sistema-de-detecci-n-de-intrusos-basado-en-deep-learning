from pathlib import Path
import pandas as pd
import random
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

input_path = Path.home() / "Documents" / "TFG" / "Testing"
output_path = Path.home() / "Documents" / "TFG" / "output"

def random_algorithm(file):
    df = pd.read_csv(file)
    labels = df['Label']
    predictions = [random.randint(0, 1) for _ in labels]

    total_benign = (labels == 0).sum()
    total_attack = (labels == 1).sum()
    detected_benign = sum((labels == 0) & (predictions == 0))
    detected_attack = sum((labels == 1) & (predictions == 1))

    return labels, predictions, total_benign, total_attack, detected_benign, detected_attack


def always_true(file):
    df = pd.read_csv(file)
    labels = df['Label']
    predictions = [1 for _ in labels]

    total_benign = (labels == 0).sum()
    total_attack = (labels == 1).sum()
    detected_benign = sum((labels == 0) & (predictions == 0))
    detected_attack = sum((labels == 1) & (predictions == 1))

    return labels, predictions, total_benign, total_attack, detected_benign, detected_attack


def always_false(file):
    df = pd.read_csv(file)
    labels = df['Label']
    predictions = [0 for _ in labels]

    total_benign = (labels == 0).sum()
    total_attack = (labels == 1).sum()
    detected_benign = sum((labels == 0) & (predictions == 0))
    detected_attack = sum((labels == 1) & (predictions == 1))

    return labels, predictions, total_benign, total_attack, detected_benign, detected_attack


def compute_metrics(labels, predictions):
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, zero_division=1)
    recall = recall_score(labels, predictions)
    f1 = f1_score(labels, predictions)
    return accuracy, precision, recall, f1


if name == 'main':
    output = open("results.txt", "w")

    output.write(f"Always True\n")

    for file in Path(input_path).rglob('*.csv'):
        labels, predictions, total_benign, total_attack, detected_benign, detected_attack = always_true(file)
        accuracy, precision, recall, f1 = compute_metrics(labels, predictions)
        print(f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        output.write(f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")

    output.write(f"\nTotal Benign: {total_benign}\n")
    output.write(f"Total Attack: {total_attack}\n")
    output.write(f"Detected Benign: {detected_benign}\n")
    output.write(f"Detected Attack: {detected_attack}\n")

    output.write(f"Always False\n")

    for file in Path(input_path).rglob('*.csv'):
        labels, predictions, total_benign, total_attack, detected_benign, detected_attack = always_false(file)
        accuracy, precision, recall, f1 = compute_metrics(labels, predictions)
        print(f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        output.write(f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")

    output.write(f"Random\n")

    for file in Path(input_path).rglob('*.csv'):
        labels, predictions, total_benign, total_attack, detected_benign, detected_attack = random_algorithm(file)
        accuracy, precision, recall, f1 = compute_metrics(labels, predictions)
        print(
            f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
        output.write(
            f"File: {file.name}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}\n")

    output.close()