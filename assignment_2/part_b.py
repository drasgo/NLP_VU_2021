import spacy
import pprint

nlp = spacy.load("en_core_web_sm")
text = open("datasets/conllst.2017.trial.simple.conll", "r").read()
text = text.split("\n")
phrases = []
phrase = []
index = 0
for tok in text:
    if not tok.replace(" \t", ""):
        continue
    token = tok.split("	")
    if int(token[0]) != index + 1:
        phrases.append(" ".join(phrase))
        phrase = []

    index = int(token[0])
    phrase.append(token[1])

for phrase in phrases:
    doc = nlp(phrase)
    for token in doc:
        print(token.i+1, token.text, token.lemma_, token.tag_, [elem.i for elem in doc if elem.text == str(token.head)][0]+1,
              token.dep_)
