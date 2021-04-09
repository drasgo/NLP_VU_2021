import csv
import statistics as st
import pandas as pd
import spacy
from collections import Counter

df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
nlp = spacy.load("en_core_web_sm")
tweets = df["Tweet text"]                   #convert to dataframe with only tweets
print(tweets.head())

### 0) pre-processing steps
tweets = df["Tweet text"]                   #change 'series'  to strings
for i in df["Tweet text"]:
    tweets_token += str(i)

doc = nlp(tweets_token)                     #put it in a doc

sentences = doc.sents
for sentence in sentences:
    print()
    print(sentence)
    for token in sentence:
        print(token.text)

#doc = nlp.pipe(sentences, disable = ["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])


### Question 1. Tokanization
word_frequencies = Counter()

for sentence in doc.sents:
    words = []
    for token in sentence:
        # filter out punctuation
        if not token.is_punct:
            words.append(token.text)
    word_frequencies.update(words)

print(word_frequencies)

num_tokens = len(doc)
num_words = sum(word_frequencies.values())
num_types = len(word_frequencies.keys())

print(num_tokens, num_words, num_types)

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


### Question 2. Part of Speech Tagging

for token in tweets[0:10]:
    print(token.text, token.pos_)



### Question 3 Lemmanization

# English pipelines include a rule-based lemmatizer
#nlp = spacy.load("en_core_web_sm")
#lemmatizer = nlp.get_pipe("lemmatizer")
#print(lemmatizer.mode)  # 'rule'

#doc = nlp("I was reading the paper.")
#print([token.lemma_ for token in doc])



df = pd.read_csv('datasets/train/SemEval2018-T3-train-taskB.txt', delimiter="\t", quoting=csv.QUOTE_NONE)
nlp = spacy.load("en_core_web_sm")              #the model we're going to use
tweets = df["Tweet text"]                       #make a dataframe with only relevant colum 'tweet text'
print(tweets.head())

### Question 1. Tokanization

# Sum of tokens
print('Before PreProcessing n_Tokens: ', len(tweets))

# Sum of number of types ( = 'unique words')
cleaned_doc = []
for token in tweets:
    if not (token.is_stop):
        cleaned_doc.append(token.text)

print('Number of types are: ' (cleaned_doc))

all_stopwords = nlp.Defaults.stop_words
all_stopwords

# Sum of words
cleaned_doc1 = []
for token in tweets:
    if not (token.is_punct):
        cleaned_doc1.append(token.text)

print('After PreProcessing n_Tokens: ', len(cleaned_doc1))

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

### Question 5. Named Entity Recognition first three tweets
import spacy
from spacy import displacy

text = ("Sweet United Nations video. Just in time for Christmas." 
        "We are rumored to have talked to Erv's agent,and the Angels asked about Ed Escobar,that's hardly nothing"
        "Hey there! Nice to see you ")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)

displacy.render(doc, style = "ent", jupyter= True)

spacy.explain('ORG')

### Question 2. Part of Speech Tagging

for token in tweets[0:10]:
    print(token.text, token.pos_)



### Question 3 Lemmanization

# English pipelines include a rule-based lemmatizer
#nlp = spacy.load("en_core_web_sm")
#lemmatizer = nlp.get_pipe("lemmatizer")
#print(lemmatizer.mode)  # 'rule'

#doc = nlp("I was reading the paper.")
#print([token.lemma_ for token in doc])
# ['I', 'be', 'read', 'the', 'paper', '.']






### Question 5. Named Entity Recognition first three tweets
from spacy import displacy

text = ("Sweet United Nations video. Just in time for Christmas." 
        "We are rumored to have talked to Erv's agent,and the Angels asked about Ed Escobar,that's hardly nothing"
        "Hey there! Nice to see you ")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)

displacy.render(doc, style = "ent", jupyter= True)

spacy.explain('ORG')

