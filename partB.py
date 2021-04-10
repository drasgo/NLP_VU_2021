import pandas as pd
import csv
import random
import numpy as np
import sklearn.metrics as sk

# B - Class distributions
df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)

tweets = df["Tweet text"]
labels = df["Label"]

class_count = labels.value_counts()
class_freq = class_count / sum(class_count)
print("The number of instances per class: ", class_count)
print("The relative frequencies per class: ", class_freq, '\n')

for i in range(4):
    # idx = np.where(labels == i)[0][0]
    print(df.loc[df.Label == i].sample(1)[['Tweet text', 'Label']])
    # print("Example sentence for class", i, ":")
    # print(tweets[idx], "\n")

# B - Baselines
random.seed(10)
# df = pd.read_csv('datasets/test_TaskB/SemEval2018-T3_input_test_taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
# tweets = df["tweet text"]
# labels = df["Label"]

# Random baseline
results_random = {
    0: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    1: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    2: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    3: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
}

metrics = ['precision', 'recall', 'f1-score']

for i in range(10):
    random_labels = np.random.randint(0, 4, len(labels))

    temp_result = sk.classification_report(labels, random_labels, output_dict=True)

    for c in range(4):
        idx = np.where(labels == c)[0]
        compare = labels[idx] == random_labels[idx]
        accuracy = sum(compare) / len(compare)
        results_random[c]['accuracy'].append(accuracy)
        for m in metrics:
            results_random[c][m].append(temp_result[str(c)][m])

metrics += ['accuracy']
for c in range(4):
    for m in metrics:
        results_random[c][m] = np.mean(results_random[c][m])
print(results_random)

# Majority baseline

major_labels = np.full_like(labels, np.argmax(class_count))

results_majority = {
    0: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    1: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    2: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    3: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
}

metrics = ['precision', 'recall', 'f1-score']

temp_result = sk.classification_report(labels, major_labels, output_dict=True, zero_division=0)

for c in range(4):
    idx = np.where(labels == c)[0]
    compare = labels[idx] == random_labels[idx]
    accuracy = sum(compare) / len(compare)
    results_majority[c]['accuracy'].append(accuracy)
    for m in metrics:
        results_majority[c][m].append(temp_result[str(c)][m])

metrics += ['accuracy']
for c in range(4):
    for m in metrics:
        results_majority[c][m] = np.mean(results_majority[c][m])
print(results_majority)

print(df.loc[df.Label == 0].sample(25)[['Tweet text', 'Label']])
