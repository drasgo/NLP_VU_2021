import pandas as pd
import csv
import random
import numpy as np
import sklearn.metrics as sk
import pprint

# B - Class distributions
print('\n------------ B1 ------------\n')
df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)

tweets = df["Tweet text"]
labels = df["Label"]

class_count = labels.value_counts()
class_freq = class_count / sum(class_count)
print("The number of instances per class:\n", class_count, '\n')
print("The relative frequencies per class:\n", class_freq, '\n')

for i in range(4):
    print("Example sentence for class: ", i)
    print(df.loc[df.Label == i].sample(1)[['Tweet text', 'Label']], '\n')

# B - Baselines
print('\n------------ B2 ------------\n')

random.seed(10)
df = pd.read_csv('datasets/goldtest_TaskB/SemEval2018-T3_gold_test_taskB_emoji.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
tweets = df["Tweet text"]
labels = df["Label"]

class_count = labels.value_counts()
class_freq = class_count / sum(class_count)
print("The number of instances per class:\n", class_count, '\n')
print("The relative frequencies per class:\n", class_freq, '\n')

# Random baseline
results_random = {
    0: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    1: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    2: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    3: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
}

metrics = ['precision', 'recall', 'f1-score']

for i in range(100):
    random_labels = np.random.randint(0, 4, len(labels))
    temp_result = sk.classification_report(labels, random_labels, output_dict=True)
    accuracies = sk.confusion_matrix(labels, random_labels, normalize="true").diagonal()

    # Insert metric scores into a dictionary
    for c in range(4):
        results_random[c]['accuracy'].append(accuracies[c])
        for m in metrics:
            results_random[c][m].append(temp_result[str(c)][m])

# Generate final dict with results
# Calculate and round the metrics
# Calculate macro and weighted averages per metric
metrics += ['accuracy']
for m in metrics:
    metric_results = []
    for c in range(4):
        results_random[c][m] = round(np.mean(results_random[c][m]), 2)
        metric_results.append(results_random[c][m])
    results_random['macro_' + m] = round(np.mean(metric_results), 2)
    results_random['weighted_' + m] = round(np.average(metric_results, weights=class_freq.values), 2)

print('Dictionary of random baseline')
pprint.pprint(results_random)

# Majority baseline

results_majority = {
    0: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    1: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    2: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []},
    3: {'accuracy': [], 'precision': [], 'recall': [], 'f1-score': []}
}

metrics = ['precision', 'recall', 'f1-score']
major_labels = np.full_like(labels, np.argmax(class_count))
temp_result = sk.classification_report(labels, major_labels, output_dict=True, zero_division=0)
accuracies = sk.confusion_matrix(labels, major_labels, normalize="true").diagonal()
for c in range(4):
    results_majority[c]['accuracy'].append(accuracies[c])
    for m in metrics:
        results_majority[c][m].append(temp_result[str(c)][m])

# Generate final dict with results
# Calculate and round the metrics
# Calculate macro and weighted averages per metric
metrics += ['accuracy']
for m in metrics:
    metric_results = []
    for c in range(4):
        results_majority[c][m] = round(np.mean(results_majority[c][m]), 2)
        metric_results.append(results_majority[c][m])
    results_majority['macro_' + m] = round(np.mean(metric_results), 2)
    results_majority['weighted_' + m] = round(np.average(metric_results, weights=class_freq.values), 2)

print('\nDictionary of majority baseline')
pprint.pprint(results_majority)
