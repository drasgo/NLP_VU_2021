import spacy
from spacy.tokens.doc import Doc

def clean_up_sentence(sent: list) -> str:
    """ Cleans up the sentence, removing all the spaces obtained by the tokenization"""
    return " ".join(sent).replace(" '", "'").replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" n't", "n't")

result = []
index = 0
# Retrieves the dataset from the conll file
text = open("datasets/conllst.2017.trial.simple.conll", "r").read().split("\n")
phrase = []
phrases = {}

# Iterates through the dataset and retrieves the text dividing it in different sentences based on the index numbers. The sentences are retrieved and aggregated to form a string, which is going to be used as key, and the tokenized elements are going to be saved in a list, saved as value for that key

for token in text:
    # Split the token considering the spaces
    split_token = token.split("	")

    # Between sentences there is an empty space
    if not token.replace(" \t", ""):
        continue

    # In the dataset, the first value in each row is the index of the token, and we can use that to identify the different sentences, as each new sentence will restart the index from 1.
    if int(split_token[0]) != index + 1:
        sentence = clean_up_sentence(phrase)
        # Add the new pair key(sentence)-value(pre-tokenized sentence) to the dictionary
        phrases[sentence] = phrase
        # Reset the temporary list
        phrase = []

    index = int(split_token[0])
    # The second element in each row is the tokenized part of the sentence
    phrase.append(split_token[1])

# The last phrase is not added to the dictionary in the for, so we're manually adding it here
sentence = clean_up_sentence(phrase)
phrases[sentence] = phrase


# Initialize spacy's english small model
nlp = spacy.load("en_core_web_sm")

def tokenizer(txt: str) -> Doc:
    """
    This function overrides the spacy tokenizer, because we are using the already tokenized phrases retrieved from the dataset.
    It creates and returns a Doc object containing the already tokenized sentence, which is going to go through the remaining of the modules in the standard pipeline.
    Basically it skips spacy's tokenization
    """
    if txt in phrases:
        return Doc(nlp.vocab, phrases[txt])
    else:
        raise ValueError('No tokenization available for input.')

# Dictionary that maps spacy labels to golden standard labels.
translation_dict = {
    'ROOT': 'root'
}

def transform_labelset(token, dict):
    """"
    This function tries to map the label from spacy label set to a label from the golden standard label set.
    """
    if token in dict:
        return dict[token]
    else:
        return token


# Overrides spacy's tokenizer with our function
nlp.tokenizer = tokenizer
# Run spacy for each sentence and save the results in the "result" list (which is later going to be saved in a file)
# NOTE: even if we are passing our normal string sentence, the tokenizer is overridden so we actually use the tokenization retrieved from the dataset.
for ph in phrases:
    parser = nlp(ph)

    for token in parser:
    # For consistency reasons, the information extracted is the same (and in the same order) as the one available in the file conllst.2017.trial.simple.dep.conll
        result.append(f"{token.i+1}	"
                      f"{token.text}	"
                      f"{token.lemma_}	"
                      f"{token.tag_}	"
                      f"{[elem.i for elem in parser if elem.text == str(token.head)][0]+1}	"
                      f"{transform_labelset(token.dep_, translation_dict)}")
    result.append(" ")

# Join the results on multiple lines and save the final output
open("datasets/cleaned_conllst_2017_trial_simple_conll", "w").write("\n".join(result))
print("Done!")
