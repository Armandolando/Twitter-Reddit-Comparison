import bz2, json, os, gzip, numpy
from langdetect import detect
import numpy as np
from nltk.tokenize import RegexpTokenizer
from compute_embeddings import get_stopwords, save_embeddings



def filter_tweets(path):
    """
    Takes tweets from a specific day and return only the ones in english language
    :param path: path of the tweets (up to the day of publishment)
    :return: list of tweets in a specific day, filtered by english language
    """
    hours_directories = os.listdir(path) # getting directories associated to different hours
    en_tweets = []
    for i, hour in enumerate(hours_directories):
        print('Fetching from hour directory: ' + str(i+1) + ' of ' + str(len(hours_directories)))
        files_path = path + '\\' + hour
        files = os.listdir(files_path)
        for j, file in enumerate(files):
            print('Fetching files: ' + str(j+1) + ' of ' + str(len(files)))
            f = bz2.BZ2File(files_path + '\\' + file, "r")
            for line in f:
                tweet = json.loads(line)
                if 'delete' not in tweet:
                    try:
                        language = detect(tweet['text'])
                    except:
                        continue
                    if language == 'en':
                        en_tweets.append(tweet['text'])
    return en_tweets


def save_tweets(filtered_tweets):
    """
    Compresses and saves the filtered tweets
    :param filtered_tweets: list of tweets to be saved
    """
    f = gzip.GzipFile("en_tweets.npy.gz", "w")
    numpy.save(file=f, arr=filtered_tweets)
    f.close()

def preprocess_tweets(tweets):
    tokenizer = RegexpTokenizer(r'\w+')
    stopwords = get_stopwords()
    processed_tweets = []
    for i, tweet in enumerate(tweets):
        words = tokenizer.tokenize(tweet)  # getting rid of punctuation
        words = [w.lower() for w in words]  # applying low case
        words = [w for w in words if w not in stopwords]  # applying stopwords
        processed_tweets.append(' '.join(words))
        if i == 200000:
            break
    return processed_tweets

if __name__ == '__main__':
    tweets_path = r'C:\Users\zippo\Desktop\twitter\2015\03\22'
    #filtered_tweets = filter_tweets(tweets_path)
    #save_tweets(filtered_tweets)

    path = "en_tweets.npy.gz"
    f = gzip.GzipFile(path, "r")
    tweets = np.load(f)

    p_tweets = preprocess_tweets(tweets)

    print(len(p_tweets))





