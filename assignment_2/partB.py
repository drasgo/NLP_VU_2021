import spacy

nlp = spacy.load("en_core_web_sm")
text = open("datasets/conllst.2017.trial.simple.conll", "r").read()
text = text.split("\n")
phrases = []
phrase = []
index = 0

# Extract information from the given file
for tok in text:
    if not tok.replace(" \t", ""):
        continue
        
    token = tok.split("	")
    if int(token[0]) != index + 1:
        phrases.append(" ".join(phrase))
        phrase = []

    index = int(token[0])
    phrase.append(token[1])

result = []
# Runs spacy for every sentence, and save the output in the "result" list
for phrase in phrases:
    doc = nlp(phrase)
    for token in doc:
        print(token.i+1, token.text, token.lemma_, token.tag_, [elem.i for elem in doc if elem.text == str(token.head)][0]+1,
              token.dep_)
        result.append(f"{token.i+1}	"
                      f"{token.text}	"
                      f"{token.lemma_}	"
                      f"{token.tag_}	"
                      f"{[elem.i for elem in doc if elem.text == str(token.head)][0]+1}	"
                      f"{token.dep_}")

open("cleaned_conllst_2017_trial_simple_conll", "w").write("\n".join(result))
print("Done!")
