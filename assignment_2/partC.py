import re
import pprint
import pandas as pd

# --------------- Loading the files into lists ---------------

output = open("datasets/cleaned_conllst_2017_trial_simple_conll", "r").read()
output = output.split("\n")

sentences_output = []
sentence = []
for word in output:
    if word == " ":
        sentences_output.append(sentence)
        sentence = []

    else:
        word = re.split(r'\t', word)
        sentence.append(word)

golden = open("datasets/conllst.2017.trial.simple.dep.conll", "r").read()
golden = golden.split("\n")

sentences_gold = []
sentence = []
for word in golden:
    if word == "":
        sentences_gold.append(sentence)
        sentence = []

    else:
        word = re.split(r'\t', word)
        sentence.append(word)


# --------------- Q8 Parser 2 Gold ---------------

# If the two files have the same taken numbers, then all sentences should have the same number of tokens
lengths_output = [len(i) for i in sentences_output]
print('The number of tokens per sentence for the output file: \n', lengths_output)

lengths_golden = [len(i) for i in sentences_gold]
print('\nThe number of tokens per sentence for the golden file: \n', lengths_golden)

print('\nAll sentences have the same number of tokens:', lengths_golden == lengths_output)

# Does every sentence have a root node?
root_check = []
for sentence in sentences_output:
    nodes = [i[-1] for i in sentence]
    if 'root' in nodes:
        root_check.append('True')
    else:
        root_check.append('False')

print("\nDoes the parser find a root node for every sentence?\n", root_check)

# Is the token assigned the same dependency head/label
df_compare = pd.DataFrame(columns=['TokenID', 'Token#', 'Word', 'POS', 'Out_head', 'Gold_head', 'Out_label', 'Gold_label'
                                  , 'Out_deps', 'Gold_deps'])

tokenID = 1
for i, sentence in enumerate(sentences_output):
    for j, word in enumerate(sentence):

        # Create a list with dependent children for the word in the output file
        output_dep = [w[0] for w in sentence if w[4] == word[0]]

        # Create a list with dependent children for the word in the golden file
        golden_dep = [w[0] for w in sentences_gold[i] if w[4] == word[0]]

        # Add the token with information from both files to a pandas dataframe
        entry = {
            'TokenID': tokenID,
            'Token#': word[0],
            'Word': word[1],
            'POS': word[3],
            'Out_head': word[4],
            'Gold_head': sentences_gold[i][j][4],
            'Out_label': word[5],
            'Gold_label': sentences_gold[i][j][5],
            'Out_deps': output_dep,
            'Gold_deps': golden_dep
        }
        df_compare = df_compare.append(entry, ignore_index=True)
        tokenID += 1

# Create a second dataframe that only contains tokens with at least 1 error.
df_errors = df_compare[(df_compare['Out_head'] != df_compare['Gold_head']) | (df_compare['Out_label'] !=
                        df_compare['Gold_label']) | (df_compare['Out_deps'] != df_compare['Gold_deps'])]

# insert the first 50 errors into a csv-file for a latex table generator
df_errors.head(50).to_csv(r'./errors.csv', index=False)

# --------------- Q9 Error POS ---------------

# Create dictionary to count how often each POS tag occurs
count_dict = {}
# Create dictionary to count how often each POS tag is mislabeled
error_dict = {}
for i, sentence in enumerate(sentences_gold):
    for j, token in enumerate(sentence):
        if token[3] in count_dict:
            count_dict[token[3]] += 1
        else:
            count_dict[token[3]] = 1
            error_dict[token[3]] = 0
        if token != sentences_output[i][j]:
            error_dict[token[3]] += 1

# Calculate the percentage of error
error_pos = {}
for value in list(count_dict):
    error_pos[value] = round(error_dict[value] / count_dict[value], 3)

# Display the resulting error percentages per POS dictionary descending
error_pos = dict(sorted(error_pos.items(), key=lambda item: item[1], reverse=True))
print('\nSorted dictionary that displays the percentage of error per POS tag')
pprint.pprint(error_pos, sort_dicts=False)

# insert dictionary into a csv-file for a latex table generator
with open('pos_errors.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in error_pos.items()]

# --------------- Q10 Error Label ---------------
# Create dictionary to count how often each dependency label occurs
count_dict = {}
# Create dictionary to count how often each dependency label is mislabeled
error_dict = {}
for i, sentence in enumerate(sentences_gold):
    for j, token in enumerate(sentence):
        if token[5] in count_dict:
            count_dict[token[5]] += 1
        else:
            count_dict[token[5]] = 1
            error_dict[token[5]] = 0

        if token != sentences_output[i][j]:
            error_dict[token[5]] += 1

# Calculate the percentage of error
error_label = {}
for value in list(count_dict):
    error_label[value] = round(error_dict[value] / count_dict[value], 3)

# Display the resulting error percentages per label dictionary descending
error_label = dict(sorted(error_label.items(), key=lambda item: item[1], reverse=True))
print('Sorted dictionary that displays the percentage of error per dependency label')
pprint.pprint(error_label, sort_dicts=False)

# insert dictionary into a csv-file for a latex table generator
with open('label_errors.csv', 'w') as f:
    [f.write('{0},{1}\n'.format(key, value)) for key, value in error_label.items()]