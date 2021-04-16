import csv
import statistics as st
import pandas as pd
import spacy
from collections import Counter
from spacy import displacy

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
frequencies = Counter(word.frequencies())

print('the number of tokens is:  ', (num_tokens))
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

# Make a dictionary of all POS-tags and sort them on values
result = {}

for tweet in tweets_token:
    doc = nlp(tweet)
    for word in doc:
        if word.pos_ in result:
            result[word.pos_] += 1
        else:
            result[word.pos_] = 1

total_occurences = 0

for key in result:
    total_occurences += result[key]

pos_tags = ({key: value for key, value in sorted(result.items(), key=lambda x: x[1], reverse= True)})
#print('the total occurences of all POS-tags is:  ', (pos_tags))

### Make new dictionary with the ten most used POS_tags and sort them on value
counter = 0
final_result = {}

for key in pos_tags:
    if counter < 9:
        final_result[key] = pos_tags[key]
    else:
        break
    counter += 1

ten_most = ({key: value for key, value in sorted(final_result.items(), key=lambda item: item[1], reverse= True)})
#print('the ten most used POS-tags are:  ', (ten_most))


### Compute the relative frequency
total = 0

# Calculating the sum
for i in ten_most.values():
    total = total + i
#print("\nThe Total Sum of Values : ", total)

# Calculating the relative frequency
for x in ten_most.values():
    relative_frequency = (x/total)*100

print(relative_frequency)




### Question 3) Lemmanization
for tweet in tweets_token:
    doc = nlp(tweet)
    print(tweet)
    for word in doc:
        print(word.text, word.lemma_)

    input()


### Question 5) Named Entity Recognition first three tweets
text = ("Sweet United Nations video. Just in time for Christmas." 
        "We are rumored to have talked to Erv's agent,and the Angels asked about Ed Escobar,that's hardly nothing"
        "Hey there! Nice to see you ")
doc = nlp(text)

for entity in doc.ents:
    print(entity.text, entity.label_)

displacy.render(doc, style = "ent", jupyter= True)

spacy.explain("MONEY")



