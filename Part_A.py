import csv
import pandas as pd
import spacy
from collections import Counter
from spacy import displacy
import json
import pprint

df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
nlp = spacy.load("en_core_web_sm")
tweets = df["Tweet text"]
print(tweets.head())

num_words = 0
num_tokens = 0
types_list = []
total_word_length = 0
POS_freq = Counter()
TAG_freq = Counter()
token_dict = {}

for index, tweet in df['Tweet text'].iteritems():
    doc = nlp(tweet)

    for sentence in doc.sents:
        POS_list = []
        TAG_list = []

        for token in sentence:
            num_tokens += 1
            if token.text not in types_list:
                types_list.append(token.text)

            if not token.is_punct:
                num_words += 1
                total_word_length += len(token)
                POS_list.append(token.pos_)
                TAG_list.append(token.tag_)

                if token.text not in token_dict:
                    token_dict[token.text] = {}
                    token_dict[token.text]['count'] = 0
                    token_dict[token.text]['TAG'] = token.tag_
                    token_dict[token.text]['POS'] = token.pos_

                token_dict[token.text]['count'] += 1

        POS_freq.update(POS_list)
        TAG_freq.update(TAG_list)

num_types = len(types_list)

### A1) Print the number of token, words, types:
print('the number of tokens is:  ', num_tokens)
print('the number of words is:  ', num_words)
print('the number of types is:  ', num_types)
print('the average number of words per tweet: ',num_words/len(df))
print('the average word length: ', total_word_length/num_words)

### A2) POS-TAGGING
df = pd.DataFrame.from_dict(token_dict, orient='index').reset_index()       #make a dataframe of the dictionary
ten_most_POS = POS_freq.most_common(10)
ten_most_TAG = TAG_freq.most_common(10)
print(ten_most_POS)
print(ten_most_TAG)

# get the relative frequency
with open("t.json", "r") as fp:
    final_result = json.load(fp)

total = sum(final_result.values())
freq = {}
for pos in final_result:
    freq[pos] = final_result[pos] / total * 100
pprint.pprint(freq)

# get the most frequent used words
z = df[df['TAG'] == 'NNP'].sort_values(by='count', ascending=False).head(3)
print(z)

# get the least frequent used word
x = df[df['TAG'] == 'NNP'].sort_values(by='count', ascending=False).tail(1)
print(x)


## A3) Lemmanization
df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)

for index, tweet in df['Tweet text'].iloc[:50].iteritems():         # lemma's of first 50 tweets
    doc = nlp(tweet)
    for tweets in doc.sents:
        for token in tweets:
            if token.text != token.lemma_:
                print(token.text, token.lemma_, tweets)

### A5) Named Entity
text = ("Sweet United Nations video. Just in time for Christmas."
        "We are rumored to have talked to Erv's agent,and the Angels asked about Ed Escobar,that's hardly nothing"
        "Hey there! Nice to see you ")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)
    displacy.render(doc, jupyter=True, style='ent')
