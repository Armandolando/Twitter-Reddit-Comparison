import pandas as pd
from datasets import load_dataset
from transformers import BertTokenizer
from torch.utils.data import TensorDataset
from transformers import BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup
import numpy as np
from sklearn.metrics import f1_score, confusion_matrix, classification_report
import random
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import pymongo
from nltk.corpus import stopwords
import os
from tqdm import trange
import json
from collections import Counter, OrderedDict

def plot_positivity(reddit, twitter, topics):
    barWidth = 0.25

    # set heights of bars
    formal_reddit_bars = []
    informal_reddit_bars = []
    formal_twitter_bars = []
    informal_twitter_bars = []

    for topic in topics:
        reddit_labels = Counter(reddit[topic])
        formal_reddits = reddit_labels[1]
        informal_reddits = reddit_labels[0]
        formal_reddit_bars.append(formal_reddits / (formal_reddits + informal_reddits))
        informal_reddit_bars.append(informal_reddits / (formal_reddits + informal_reddits))
        twitter_labels = Counter(twitter[topic])
        formal_tweets = twitter_labels[1]
        informal_tweets = twitter_labels[0]
        formal_twitter_bars.append(formal_tweets / (formal_tweets + informal_tweets))
        informal_twitter_bars.append(informal_tweets / (formal_tweets + informal_tweets))

    # Set position of bar on X axis
    ind = np.arange(len(formal_reddit_bars))  # the x locations for the groups
    width = 0.25
    plt.bar(ind, formal_reddit_bars, width, color='b', label='Positive Reddits')
    plt.bar(ind, informal_reddit_bars, width, bottom=formal_reddit_bars, color='orange', label='Negative Reddits')
    plt.bar(ind + 0.30, formal_twitter_bars, width, color='c', label='Positive Tweets')
    plt.bar(ind + 0.30, informal_twitter_bars, width, bottom=formal_twitter_bars, color='red', label='Negative Tweets')

    topics[1] = 'Human-Relationships'

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    plt.ylabel('Percentage of positive/negative posts', fontweight='bold', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(formal_reddit_bars))], topics, rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)

    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_comments_emotions(reddit_pr, twitter_pr, topic):
    """
    Plots a bar plot representing the percentage of comments for each emotion
    :param reddit_pr: list of reddit comments of a certain topic
    :param twitter_pr: list of twitter comments of a certain topic
    :param topic: name of the topic
    """
    # set width of bars
    barWidth = 0.25

    # set heights of bars
    reddit_bars = []
    twitter_bars = []
    emotions = ['Sadness', 'Joy', 'Love', 'Anger', 'Fear', 'Surprise']

    reddit_dict = dict(sorted(Counter(reddit_pr).items()))
    twitter_dict = dict(sorted(Counter(twitter_pr).items()))
    for emotion in range(len(emotions)):
        #reddit_bars.append(reddit_dict[emotion] / len(reddit_pr))
        if emotion in reddit_dict.keys():
            reddit_bars.append(reddit_dict[emotion] / len(reddit_pr))
        else:
            reddit_bars.append(0)
        if emotion in twitter_dict.keys():
            twitter_bars.append(twitter_dict[emotion] / len(twitter_pr))
        else:
            twitter_bars.append(0)

    # Set position of bar on X axis
    r1 = np.arange(len(reddit_bars))
    r2 = [x + barWidth for x in r1]

    # Make the plot
    plt.bar(r1, reddit_bars, color='orange', width=barWidth, edgecolor='white', label='Reddit')
    plt.bar(r2, twitter_bars, color='blue', width=barWidth, edgecolor='white', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Emotion', fontweight='bold')
    plt.ylabel('Normalized number of comments with a certain emotion', fontweight='bold')
    plt.xticks([r + barWidth for r in range(len(reddit_bars))], emotions)

    # Create legend & Show graphic
    plt.title('TOPIC: ' + topic)
    plt.grid()
    plt.legend()
    plt.savefig(f'{topic}_sentiment.png', bbox_inches='tight')
    plt.show()

def text_cleaning(texts):
    new_texts = []
    for text in texts:
        new_text = []
        for word in text.split(' '):
            if '@' in word or 'http' in word or word == 'RT' or word in stopwords.words('english') or word in stopwords.words('italian'):
                continue
            new_text.append(word)
        new_texts.append(' '.join(new_text))
    return new_texts

def emotion_classification(dataloader, model):

    model.load_state_dict(torch.load('../emotions_analysis/Models/EM_BERT_ft_Epoch2.model'))
    model.eval()
    
    predictions, true_vals = [], []
    
    for batch in tqdm(dataloader):
        
        batch = tuple(b.to(device) for b in batch)
        
        inputs = {'input_ids':      batch[0],
                  'attention_mask': batch[1],
                 }

        with torch.no_grad():        
            outputs = model(**inputs)
            
        logits = outputs[0]

        logits = logits.detach().cpu().numpy()
        predictions.append(logits)
    
    predictions = np.concatenate(predictions, axis=0)
    preds_flat = np.argmax(predictions, axis=1).flatten()
    preds = []
    for pred in preds_flat:
        if pred in [0,3,4]:
            preds.append(0)
        else:
            preds.append(1)
    return preds

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)

labels_dict = {0:'sadness', 1:'joy', 2:'love', 3:'anger', 4:'fear', 5:'surprise'}
label_dict = {'sadness': 0, 'joy': 1, 'love': 2, 'anger': 3, 'fear': 4, 'surprise': 5}

model_fi = BertForSequenceClassification.from_pretrained(
                                      'bert-base-uncased', 
                                      num_labels = len(labels_dict),
                                      output_attentions = False,
                                      output_hidden_states = False
                                     )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_fi.to(device)

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

#topic_list = ['Music', 'Religion', 'Smartphones', 'Motors', 'Racism', 'Alcohol', 'Sexual-orientation']
topic_list = ['Smartphones', 'Sex', 'Body-care', 'Education', 'Books', 'Money', 'Racism', 'Sexual-orientation', 'Baseball', 'Fishing', 'Sleeping', 'Music']
#topic_list = ['Smartphones']

#print(topic_list)

twitter_dict = {}
reddit_dict = {}

for topic in topic_list:
    twitter_dict[topic] = []
    reddit_dict[topic] = []

for data in data_list_twitter:
    for topic, index in zip(topic_list,range(len(topic_list))):
        if topic in data.keys():
            for tweet in data[topic]:
                if not tweet['text'].startswith('RT'):
                    twitter_dict[topic].append(tweet['text'])


for data in data_list_reddit:
    for topic, index in zip(topic_list,range(len(topic_list))):
        if topic in data.keys():
            for post in data[topic]:
                if not post['body'] is None:
                    reddit_dict[topic].append(post['body'])

#print(twitter_dict['Sex'])

twitter_preds = {}
reddit_preds = {}

for topic in topic_list:
    texts = text_cleaning(twitter_dict[topic])
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data['input_ids']
    attention_masks_train = encoded_data['attention_mask']

    dataset = TensorDataset(input_ids_train, 
                                attention_masks_train,
                                )

    batch_size = 5

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size
    )
    twitter_preds[topic] = emotion_classification(dataloader, model_fi)

for topic in topic_list:
    texts = text_cleaning(reddit_dict[topic])
    encoded_data = tokenizer.batch_encode_plus(
        texts,
        add_special_tokens=True,
        return_attention_mask=True,
        pad_to_max_length=True,
        max_length=256,
        return_tensors='pt'
    )

    input_ids_train = encoded_data['input_ids']
    attention_masks_train = encoded_data['attention_mask']

    dataset = TensorDataset(input_ids_train, 
                                attention_masks_train,
                                )

    batch_size = 5

    dataloader = DataLoader(
        dataset,
        shuffle=False,
        batch_size=batch_size
    )
    reddit_preds[topic] = emotion_classification(dataloader, model_fi)

plot_positivity(reddit_preds, twitter_preds, topic_list)