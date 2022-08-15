import gzip, numpy, time, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
#import stanza
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
import os
from tqdm import trange
import json
from nltk.corpus import stopwords
import gensim.corpora as corpora
from gensim.models.coherencemodel import CoherenceModel

def intersection(lst1, lst2):
    lst3 = [value for value in lst1 if value in lst2]
    return lst3

def text_cleaning(texts, text_key):
    new_texts = []
    for text in texts:
        #print(text)
        new_text = []
        for word in text[text_key].split(' '):
            word = word.lower()
            if '@' in word or word == '&amp;' or word == 'new' or word == 'like' or 'http' in word or word == 'rt' or word in stopwords.words('english') or word in stopwords.words('italian'):
                continue
            new_text.append(word)
        new_texts.append(' '.join(new_text))
    return new_texts

def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)

    return tf_idf, count

def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in enumerate(labels)}
    return top_n_words


def top_words(docs_df):
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(docs_df['Doc']))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    return top_n_words

path_data_twitter = '../data/twitter/'
data_list_twitter = []

path_data_reddit = '../data/reddit/'
data_list_reddit = []

for filename, _ in zip(os.listdir(path_data_twitter), trange(len(os.listdir(path_data_twitter)))):
    file_ = open(path_data_twitter+filename,)
    data_list_twitter.append(json.load(file_))

for filename, _ in zip(os.listdir(path_data_reddit), trange(len(os.listdir(path_data_reddit)))):
    file_ = open(path_data_reddit+filename,)
    data_list_reddit.append(json.load(file_))


twitter_dict = {"Doc":[], "Topic":[]}
reddit_dict = {"Doc":[], "Topic":[]}


for data in data_list_twitter:
    for topic, index in zip(data.keys(),trange(len(data.keys()))):
        tweets = text_cleaning(data[topic], 'text')
        for tweet in tweets:
            if not tweet.startswith('rt'):
                twitter_dict["Doc"].append(tweet)
                twitter_dict["Topic"].append(topic)


for data in data_list_reddit:
    for topic, index in zip(data.keys(),trange(len(data.keys()))):
        posts = text_cleaning(data[topic], 'body')
        for post in posts:
            if not post is None:
                reddit_dict["Doc"].append(post)
                reddit_dict["Topic"].append(topic)

print(f'Twitter: {len(twitter_dict["Doc"])} Reddit {len(reddit_dict["Doc"])} Total: {len(reddit_dict["Doc"]) + len(twitter_dict["Doc"])}')

twitter_df = pd.DataFrame(twitter_dict)
reddit_df = pd.DataFrame(reddit_dict)

twitter_df = twitter_df.sample(frac=1)
reddit_df = reddit_df.sample(frac=1)

print(twitter_df.head())
print(reddit_df.head()) 

twitter_top = top_words(twitter_df)
reddit_top = top_words(reddit_df)

#reddit
topics_names = []
topics = []

for topic in reddit_top.keys():
    topics_names.append(topic)
    topics.append([terms[0] for terms in reddit_top[topic]])

input_data = [doc.split(' ') for doc in reddit_df['Doc']]

id2word = corpora.Dictionary(input_data)
corpus = [id2word.doc2bow(text) for text in input_data]

cm = CoherenceModel(topics=topics,texts = input_data,corpus=corpus, dictionary=id2word, coherence='c_v')
coherence = cm.get_coherence_per_topic()

print(coherence)

coherence_reddit_df = pd.DataFrame({'topic':topics_names, 'coherence':coherence})

coherence_reddit_df = coherence_reddit_df.sort_values(by=['coherence'], ascending=False)

#twitter
topics_names = []
topics = []

for topic in twitter_top.keys():
    topics_names.append(topic)
    topics.append([terms[0] for terms in twitter_top[topic]])

input_data = [doc.split(' ') for doc in twitter_df['Doc']]

id2word = corpora.Dictionary(input_data)
corpus = [id2word.doc2bow(text) for text in input_data]

cm = CoherenceModel(topics=topics,texts = input_data,corpus=corpus, dictionary=id2word, coherence='c_v')
coherence = cm.get_coherence_per_topic()

coherence_twitter_df = pd.DataFrame({'topic':topics_names, 'coherence':coherence})

coherence_twitter_df = coherence_twitter_df.sort_values(by=['coherence'], ascending=False)


coherence_twitter_df = coherence_twitter_df.loc[coherence_twitter_df['coherence'] > 0.4]
coherence_reddit_df = coherence_reddit_df.loc[coherence_reddit_df['coherence'] > 0.4]

print(coherence_twitter_df.head(50))
print(coherence_reddit_df.head(50))

print('Twitter:')
print(coherence_twitter_df['topic'].tolist())
print('Reddit:')
print(coherence_reddit_df['topic'].tolist())

print(intersection(list(coherence_twitter_df['topic']), list(coherence_reddit_df['topic'])))