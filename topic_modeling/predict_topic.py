import collections
import json, gzip, re, os, numpy, sklearn, warnings, umap, hdbscan, pickle
from operator import itemgetter

import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import LdaModel
from nltk import RegexpTokenizer
from sentence_transformers import SentenceTransformer
from compute_embeddings import get_stopwords, get_comments
from bert import stanza_stem

def get_topics_names(path):
    """
    Gets the different topic names and the associated indices and cluster labels
    :param path: path of the file containing the topics' informations
    :return: lists of topic names, indices and labels
    """
    with open(path) as f:
        content = f.readlines()
    content = content[2:]
    topics_names = [item.split()[0] for item in content]
    topics_indices = [int(item.split()[1]) for item in content]
    topics_labels = [item.split()[2] for item in content]

    return topics_names, topics_indices, topics_labels


def get_topics(data, names, indexes, threshold=0.04):
    """
    Create a proper data structure to handle topics and relative keywords (with their importance)
    The result is the following dictionary:
        - key: name of the topic
        - value: dictionary:
                    - key: keyword
                    - value: importance of the keyword

    :param data: list of different elements, each one is a list of keyword - importance pairs
    :param names: list of different topics names
    :param indexes: list of int representing the index of the topics of interest
    :param threshold: float below which the keyword is discarded
    :return: dictionary described above
    """
    topics_names = names
    topics = {}
    index_counter = 0
    for name in topics_names:
        if name not in topics:
            topics[name] = {}
        for pair in data[indexes[index_counter]]:
            if float(pair[1]) > threshold:
                if pair[0] in topics[name]:
                    topics[name].update({pair[0] : max(float(pair[1]), topics[name][pair[0]])})
                else:
                    topics[name].update({pair[0] : float(pair[1])})
        index_counter += 1
    return topics

def naive_scores(topics, comment):
    """
    Generate a list of scores, whose indexes reflect the alphabetical order of the topics
    The scores are just counters of how many times the keywords of a topic appear in a comment
    :param topics: data structure created by 'get_topics' function
    :param comment: list of words composing a comment (eventually pre processed)
    :return: list of scores of one comment
    """
    topics_names = list(topics.keys())
    topics_names.sort()
    scores = [0 for _ in topics_names]
    for i, name in enumerate(topics_names):
        for word in comment:
            if word in list(topics[name].keys()):
                scores[i] += 1
    return scores

def get_new_comments(path, N, M):
    """
    Gets comments unseen in the previous unsupervised processing
    :param path: path of Reddit corpus
    :param N: number of comments processed in the unsupervised part
    :param M: number of new comments meant to be fetched
    :return:  - comments: new comments pre processed (stopwords, no-punctuation)
              - full_comments: new comments in raw form
    """
    stopwords = get_stopwords()
    tokenizer = RegexpTokenizer(r'\w+')
    comments = []
    full_comments = []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= N + 1:
                obj = json.loads(line)  # obj now contains a dict of the data
                comment = obj['body']
                full_comments.append(comment)
                comment_no_punct = tokenizer.tokenize(comment)  # getting rid of punctuation
                lower_comment = [w.lower() for w in comment_no_punct]  # applying low case
                words = [w for w in lower_comment if w not in stopwords]  # applying stopwords
                comments.append(words)
            if i == N + M:
                break
    return comments, full_comments

def generate_files(n_comments, comments):
    """
    Generates .txt files with a user defined number of comments
    :param n_comments: number of comments to be contained in each file
    :param comments: all comments to be written
    """
    path = r'..\comments'
    name1 = r'\file_'
    num_file = str(0)
    name2 = '.txt'
    counter = 0
    j = 0
    text_file = open(path + name1 + num_file + name2, "w", encoding="utf-8")

    for i, c in enumerate(comments):
        if not i%n_comments and i != 0:
            text_file.close()
            counter += 1
            num_file = str(counter)
            j = 1
            text_file = open(path + name1 + num_file + name2, "w", encoding="utf-8")
            text_file.write('\n\nCOMMENT ' + str(j) + '\n' + c + '\n')
        else:
            j += 1
            text_file.write('\n\nCOMMENT ' + str(j) + '\n' + c + '\n')
    text_file.close()

def label_comments():
    """
    Retrieves the labels assigned to the comments
    :return: a list of labels, representing the ground truth
    """
    path = r'..\labels'
    files = os.listdir(path)
    files.sort(key=lambda f: int(re.sub('\D', '', f)))
    x = []

    for file in files:
        text_file = open(path + '\\' + file, "r", encoding="utf-8")
        x.append([line.rstrip() for line in text_file])
        text_file.close()

    labels = list(numpy.concatenate(x).flat)
    return labels

def naive_predictions(comments, labels, topics2, topics_names):
    """
    Performs a prediction of the comments' labels based on the 'naive_scores'
    :param comments: list of comments pre processed
    :param labels: ground truth
    :param topics2: data structure created by 'get_topics' function
    :param topics_names: list of possible topics
    :return: list of predicted labels
    """
    predicted_labels = []
    topics_names.sort()
    for i, c in enumerate(comments):
        scores = naive_scores(topics2, c)
        if sum(scores) == 0:
            predicted_labels.append('Other')
        else:
            max_index = scores.index(max(scores))
            predicted_labels.append(topics_names[max_index])

    matches = np.array(predicted_labels) == np.array(labels)
    acc = sum(matches) / len(matches)

    return predicted_labels, acc

def ordered_topics(topics_names):
    """
    Removes repetitions and sorts the topics in alphabetical order
    :param topics_names: list of topics
    :return: ordered list of topics with no repetitions
    """
    classes = list(set(topics_names))
    classes.append('Other')
    classes.sort()

    return classes

def get_confusion_matrix(classes, truth, predictions):
    """
    Computes the confusion matrix, having on the rows the actual labels, on the columns the predicted ones
    :param classes: list of possible labels
    :param truth: ground truth
    :param predictions: predicted labels
    :return: confusion matrix in array form (cm) and pandas Dataframe form (df_cm)
    """
    warnings.filterwarnings("ignore")
    cm = sklearn.metrics.confusion_matrix(truth, predictions, classes)
    df_cm = pd.DataFrame(cm, index=[i for i in classes],
                         columns=[i for i in classes])

    return cm, df_cm

def get_class_accuracy(topics_names, cm):
    """
    Computes the percentage of correct classification for each label
    Keeps track also of the number of sample for each label
    :param topics_names: list of topics
    :param cm: confusion matrix (array form)
    :return: dictionary with the following structure:
                  - key: topic label
                  - value: list of 2 values, respectively the accuracy for that label and the number of samples
    """
    accuracies = {}
    classes = ordered_topics(topics_names)
    for i, name in enumerate(classes):
        cm_row = cm[i, :]
        n_samples = sum(cm_row)
        if n_samples != 0:
            correctness = cm_row[i] / n_samples
            accuracies[name] = [correctness, n_samples]
        else:
            accuracies[name] = [0, 0]

    return accuracies

def keyword_scores(topics, comment):
    """
    Computes the scores of each topic, keeping into account the importance of the keywords
    The indices reflect the alphabetical order of the topics labels
    :param topics: list of topics
    :param comment: list of words composing a comment
    :return: list of scores for that comment
    """
    topics_names = list(topics.keys())
    topics_names.sort()
    scores = [0. for _ in topics_names]
    for i, name in enumerate(topics_names):
        for word in comment:
            if word in list(topics[name].keys()):
                scores[i] += topics[name][word]
    return scores

def keyword_prediction(comments, topics2, topics_names, threshold=0):
    """
    Performs a prediction of the comments' labels, based on the 'keyword_scores' approach
    :param comments: list of all comments
    :param topics2: data structure created by 'get_topics' function
    :param topics_names: list of topic labels
    :return: list of predictions and their accuracy
    """
    predicted_labels = []
    topics_names = list(set(topics_names))
    topics_names.sort()
    for i, c in enumerate(comments):
        scores = keyword_scores(topics2, c)
        if sum(scores) <= threshold:
            predicted_labels.append('Other')
        else:
            max_index = scores.index(max(scores))
            predicted_labels.append(topics_names[max_index])

    return predicted_labels


def cluster_prediction(neighbors, components):
    """
    Associates the new comments to a cluster (between the ones pre-computed in the unsupervised part)
    :param neighbors: umap parameter
    :param components: umap parameter
    """
    with open('clusters.pickle', 'rb') as handle:
        cluster = pickle.load(handle)

    f = gzip.GzipFile('new_embeddings.npy.gz', "r")
    embeddings = np.load(f)
    print("\nDimensionality reduction...")
    test_items = umap.UMAP(n_neighbors=neighbors,  # small value: more clusters; large value: less clusters (VERY SENSIBLE)
                            n_components=components,  # small value: more clusters; large value: less clusters
                            metric='cosine').fit_transform(embeddings)
    print("Reduction completed")
    test_labels, strengths = hdbscan.approximate_predict(cluster, test_items)

    print(collections.Counter(test_labels))

def compute_new_embeddings(new_comments):
    """
    Computes the embeddings of comments never seen before
    :param new_comments: list of new comments
    """

    print("\nAcquiring pre-trained model...")
    model = SentenceTransformer('distilbert-base-nli-mean-tokens')
    print("Acquired")

    # Sentences are encoded by calling model.encode()
    print("\nGetting embeddings from pre-trained model...")
    embeddings = model.encode(new_comments)
    print("Embeddings calculated")

    # Let's save the embeddings
    print('\nSaving the embeddings...')
    f = gzip.GzipFile("new_embeddings.npy.gz", "w")
    numpy.save(file=f, arr=embeddings)
    f.close()
    print('Embeddings saved')

def get_comment_lda(comment):
    common_dictionary = Dictionary(comment)
    common_corpus = [common_dictionary.doc2bow(text) for text in comment]
    # Train the model on the corpus.
    N = 1  # number of different topics
    lda = LdaModel(common_corpus, num_topics=N)

    # Print the words of the different topics
    key_list = list(common_dictionary.token2id.keys())
    val_list = list(common_dictionary.token2id.values())
    for i in range(N):
        print("TOPIC " + str(i))
        for j, pair in enumerate(lda.get_topic_terms(i, 20)):
            print("Word " + str(j + 1) + ": " + key_list[val_list.index(pair[0])] +
                  " - Probability: " + str(pair[1]))
        print("\n\n")

def show_predictions(predictions, topic_name, full_comments):
    """
    Shows the full comments of a certain topic
    :param predictions: list of predictions for the comment at the same position
    :param topic_name: name of the topic of interest
    :param full_comments: list of comments not pre processed
    :return: list of comment predicted to be about a certain topic
    """
    filtered_comments = []
    for i, topic in enumerate(predictions):
        if topic == topic_name:
            print('\nComment ', i)
            print(full_comments[i])
            filtered_comments.append(full_comments[i])
    return filtered_comments

def filter_comments(comments, truth):
    filtered_comments = []
    filtered_labels = []
    for i, label in enumerate(truth):
        if label != 'Other':
            filtered_comments.append(comments[i])
            filtered_labels.append(label)

    return filtered_comments, filtered_labels



if __name__ == '__main__':
    path = "topics.npy.gz"
    path2 = r'C:\Users\zippo\Desktop\results\200k\neigh3_comp6_size40_eps017\stemming\topics.txt'
    f = gzip.GzipFile(path, "r")
    topics = np.load(f) # gets the output of the unsupervised part: list of keyword-importance pairs
    topics_names, topics_indices, topics_labels = get_topics_names(path2)
    topics_dict = get_topics(topics, topics_names,topics_indices) # dictionary with topics, keywords and relative importance

    datafile = r'..\reddit_corpus\RC_2015-01'
    truth = label_comments()
    comments, full_comments = get_new_comments(datafile, 200000, len(truth))
    #stemmed_c = stanza_stem(full_comments)

    filtered_c, filtered_l = filter_comments(comments, truth)

    #bert_comments, fc = get_comments(r'..\reddit_corpus\RC_2015-01', 205000)
    #bert_comments = bert_comments[200001:]

    #get_comment_lda([comments[0]])

    #compute_new_embeddings(full_comments)
    #cluster_prediction(3, 6)

    #predictions, accuracy = naive_predictions(comments, truth, topics_dict, topics_names)
    predictions = keyword_prediction(filtered_c, topics_dict, topics_names)

    matches = np.array(predictions) == np.array(filtered_l)
    acc = sum(matches) / len(matches)

    classes = ordered_topics(topics_names)
    cm, df_cm = get_confusion_matrix(classes, filtered_l, predictions)
    cl_acc = get_class_accuracy(topics_names, cm)
    print(cl_acc)
    print('Test accuracy: ' + str(round(acc*100, 2)) + '%')


    #_ = show_predictions(predictions, 'Amazon', full_comments)
    #print('\n\n', collections.Counter(predictions))

    print(len(predictions))
















"""
    df_cm = pd.DataFrame(x, index=[i for i in classes],
                         columns=[i for i in classes])
    plt.figure(figsize=(10, 7))
    sn.heatmap(df_cm, annot=True)
    plt.show()
"""

