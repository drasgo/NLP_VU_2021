import csv
import statistics as st
import pandas as pd
import spacy
from collections import Counter

df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
nlp = spacy.load("en_core_web_sm")
tweets = df["Tweet text"]
print(tweets.head())

### 0) pre-processing steps
tweets_token = [tweet for tweet in df["Tweet text"]]

for tweet in tweets_token:
    doc = nlp(tweet)
    print(tweet)
    for word in doc:
       print(word.text, word.pos_)

input()

### Question 1) Tokanization
word_frequencies = Counter()

for tweet in tweets_token:
    doc = nlp(tweet)
    words = []
    for token in doc:
        if not token.is_punct:
            words.append(token.text)
    word_frequencies.update(words)

print(word_frequencies)

num_tokens = len(doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())

print('the number of tokens is:  ', (num_token))
print('the number of words is:  ', (num_words))
print('the number of types is:  ', (num_types))

# Average tokens and word length
tweet_lengths = []
word_lengths = []

for tweet in tweets:
    tokenized = nlp(tweet)
    tweet_lengths.append(len(tokenized))
    for word in tokenized:
        word_lengths.append(len(word))

print('The average number of words per tweet:  ', st.mean(tweet_lengths))
print('The average word lenght is: ', st.mean(word_lengths))



### Question 2) Part of Speech Tagging
tweets_token = [tweet for tweet in df["Tweet text"]]

for tweet in tweets_token:
    doc = nlp(tweet)
    print(tweet)
    for word in doc:
        print(word.text,word.pos_)

    input()


### Question 3) Lemmanization

for tweet in tweets_token:
    doc = nlp(tweet)
    print(tweet)
    for word in doc:
        print(word.text, word.lematizer_)

    input()


### Question 5) Named Entity Recognition first three tweets
from spacy import displacy

text = ("Sweet United Nations video. Just in time for Christmas." 
        "We are rumored to have talked to Erv's agent,and the Angels asked about Ed Escobar,that's hardly nothing"
        "Hey there! Nice to see you ")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)

displacy.render(doc, style = "ent", jupyter= True)

spacy.explain('ORG')



