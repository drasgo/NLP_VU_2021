import spacy
from spacy.tokens.doc import Doc

nlp = spacy.load("en_core_web_sm")
text = open("datasets/conllst.2017.trial.simple.conll", "r").read()
text = text.split("\n")
phrases = {}
phrase = []
result = []
index = 0

# Iterates through the dataset and retrieves the text dividing it in different sentences based on the index numbers
# The sentences are retrieved and aggregated to form a string, which is going to be used as key, and the tokenized
#  elements are going to be saved in a list, saved as value for that key
for tok in text:
    if not tok.replace(" \t", ""):
        continue

    token = tok.split("	")
    if int(token[0]) != index + 1:
        phr = " ".join(phrase).replace(" '", "'").replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" n't", "n't")
        phrases[phr] = phrase
        phrase = []

    index = int(token[0])
    phrase.append(token[1])

# The last phrase is not added to the dictionary in the for, so we're manually adding it here
phr = " ".join(phrase).replace(" '", "'").replace(" .", ".").replace(" ,", ",").replace(" ;", ";").replace(" n't", "n't")
phrases[phr] = phrase

def tokenizer(txt: str) -> Doc:
    """
    This function overrides the spacy tokenizer, because we are using the already tokenized phrases retrieved from the dataset.
    It creates and returns a Doc object containing the already tokenized sentence, which is going to go through the
    remaining of the modules in the standard pipeline. Basically it skips spacy's tokenization
    """
    if txt in phrases:
        return Doc(nlp.vocab, phrases[txt])
    else:
        raise ValueError('No tokenization available for input.')

# Overrides spacy's tokenizer with our function
nlp.tokenizer = tokenizer
# Run spacy for each sentence and save the results in the "result" list (which is later going to be saved in a file)
# NOTE: even if we are passing our normal string sentence, the tokenizer is overridden so we actually use the
# tokenization retrieved from the dataset.
for ph in phrases:
    parser = nlp(ph)

    for token in parser:
        result.append(f"{token.i+1}	"
                      f"{token.text}	"
                      f"{token.lemma_}	"
                      f"{token.tag_}	"
                      f"{[elem.i for elem in parser if elem.text == str(token.head)][0]+1}	"
                      f"{token.dep_}")
    result.append(" ")

# Join the results on multiple lines and save the final output
open("datasets/cleaned_conllst_2017_trial_simple_conll", "w").write("\n".join(result))
print("Done!")
