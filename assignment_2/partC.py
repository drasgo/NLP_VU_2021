import re
import pprint

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
results = {}
average_error_head = []
average_error_label = []
dep_dif = []
for i, sentence in enumerate(sentences_output):
    errors_head = 0
    errors_label = 0

    dependent_dict_gold = {}
    dependent_dict_output = {}

    for j, word in enumerate(sentence):
        # Check for same dependency head
        if word[4] != sentences_gold[i][j][4]:
            errors_head += 1
        # Check for same dependency label
        if word[5] != sentences_gold[i][j][5]:
            errors_label += 1

        # Check for same dependents
        if sentences_gold[i][j][4] in dependent_dict_gold:
            dependent_dict_gold[sentences_gold[i][j][4]].append(sentences_gold[i][j][0])
        else:
            dependent_dict_gold[sentences_gold[i][j][4]] = [sentences_gold[i][j][0]]

        if word[4] in dependent_dict_output:
            dependent_dict_output[word[4]].append(word[0])
        else:
            dependent_dict_output[word[4]] = [word[0]]

    # Check if the tokens have the same dependents
    if dependent_dict_gold != dependent_dict_output:
        # print('The tokens in the output has the same dependents as in the golden file')
        dep_dif.append(False)
    else:
        dep_dif.append(True)

    average_error_head.append(round(errors_head/len(sentence), 2))
    average_error_label.append(round(errors_label/len(sentence), 2))

print("\nnumber of mislabeled dependency head per sentence:\n", average_error_head)
print("\nnumber of mislabeled dependency label per sentence:\n", average_error_label)
print('\nAll tokens have the same dependents as in the gold file?\n', dep_dif)


# --------------- Q9 Error POS ---------------

count_dict = {}
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

error_pos = {}
for value in list(count_dict):
    error_pos[value] = round(error_dict[value] / count_dict[value], 3)

error_pos = dict(sorted(error_pos.items(), key=lambda item: item[1], reverse=True))
print('\nSorted dictionary that displays the percentage of error per POS tag')
pprint.pprint(error_pos, sort_dicts=False)

# --------------- Q10 Error Label ---------------

print()
count_dict = {}
error_dict = {}
for i, sentence in enumerate(sentences_gold):
    if i == 16:
        continue

    for j, token in enumerate(sentence):
        if token[5] in count_dict:
            count_dict[token[5]] += 1
        else:
            count_dict[token[5]] = 1
            error_dict[token[5]] = 0

        if token != sentences_output[i][j]:
            error_dict[token[5]] += 1

error_pos = {}
for value in list(count_dict):
    error_pos[value] = round(error_dict[value] / count_dict[value], 3)

error_pos = dict(sorted(error_pos.items(), key=lambda item: item[1], reverse=True))
print('Sorted dictionary that displays the percentage of error per dependency label')
pprint.pprint(error_pos, sort_dicts=False)