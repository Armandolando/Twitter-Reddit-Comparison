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

def plot_total_formality(reddits, tweets):
    reddit_predictions = []
    twitter_predictions = []
    topics = reddits.keys()
    for topic in reddits.keys():
        reddit_predictions.append(list(reddits[topic]))
        twitter_predictions.append(list(tweets[topic]))

    reddit_predictions = [item for sublist in reddit_predictions for item in sublist]
    twitter_predictions = [item for sublist in twitter_predictions for item in sublist]
    reddit_formal_labels = Counter(reddit_predictions)[1]
    reddit_informal_labels = Counter(reddit_predictions)[0]
    reddit_formality_rate = reddit_formal_labels / (reddit_formal_labels + reddit_informal_labels)
    twitter_formal_labels = Counter(twitter_predictions)[1]
    twitter_informal_labels = Counter(twitter_predictions)[0]
    twitter_formality_rate = twitter_formal_labels / (twitter_formal_labels + twitter_informal_labels)
    rates = []
    rates.append(reddit_formality_rate)
    rates.append(twitter_formality_rate)
    
    width = 0.15
    plt.rc('font', size=20)
    plt.bar(0.25, reddit_formality_rate, width, color='orange', label='Reddit')
    plt.bar(0.75, twitter_formality_rate, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('OSN', fontweight='bold', fontsize=20)
    plt.xlim(0, 1)
    plt.ylabel('Formality rate', fontweight='bold')
    plt.xticks([0.25, 0.75], ['Reddit', 'Twitter'])


    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    #plt.title('Rate of formality')
    plt.grid()
    plt.legend()
    plt.show()

def plot_formality(reddit, twitter, topics):
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
    plt.bar(ind, formal_reddit_bars, width, color='b', label='Formal Reddits')
    plt.bar(ind, informal_reddit_bars, width, bottom=formal_reddit_bars, color='orange', label='Informal Reddits')
    plt.bar(ind + 0.30, formal_twitter_bars, width, color='c', label='Formal Tweets')
    plt.bar(ind + 0.30, informal_twitter_bars, width, bottom=formal_twitter_bars, color='red', label='Informal Tweets')

    topics[1] = 'Human-Relationships'

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    plt.ylabel('Percentage of formal/informal comments', fontweight='bold', fontsize=20)
    plt.xticks([r + barWidth for r in range(len(formal_reddit_bars))], topics, rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)

    # Create legend & Show graphic
    plt.title('Rate of formality')
    plt.grid()
    plt.legend()
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

def formal_classification(dataloader, model):

    model.load_state_dict(torch.load('../formal_informal/Models/FI_BERT_ft_Epoch2.model'))
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
    return preds_flat

tokenizer = BertTokenizer.from_pretrained(
    'bert-base-uncased',
    do_lower_case=True
)

labels_dict = {0:'informal', 1:'formal'}
label_dict = {'informal': 0, 'formal': 1}

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

print(topic_list)

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
    twitter_preds[topic] = formal_classification(dataloader, model_fi)

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
    reddit_preds[topic] = formal_classification(dataloader, model_fi)

plot_formality(reddit_preds, twitter_preds, topic_list)
plot_total_formality(reddit_preds, twitter_preds)