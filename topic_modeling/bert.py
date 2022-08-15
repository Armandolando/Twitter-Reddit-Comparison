import umap, hdbscan, gzip, numpy, time, pickle
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import stanza
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity

from compute_embeddings import save_embeddings, get_comments
from sklearn.feature_extraction.text import CountVectorizer
def stanza_stem(full_comments):
    nlp = stanza.Pipeline('en')  # This sets up a default neural pipeline in English
    exclude = ['AUX', 'ADP', 'PUNCT', 'PRON', 'NUM', 'ADV', 'CCONJ', 'DET', 'INTJ', 'PART', 'SCONJ', 'SYM']
    clean_comments = []
    for comment in full_comments:
        clean_comment = []
        doc = nlp(comment)
        for sentence in doc.sentences:
            for word in sentence.words:
                if word.upos not in exclude:
                    clean_comment.append(word.lemma)
        result = ' '.join(clean_comment)
        clean_comments.append(result)
    return clean_comments

def load_embeddings(path):
    print('\nLoading embeddings...')
    f = gzip.GzipFile(path, "r")
    embeddings = np.load(f)
    print('Embeddings loaded')
    return embeddings

def reduce_dimension(data, neighbors, components):
    print("\nDimensionality reduction...")
    umap_embeddings = umap.UMAP(n_neighbors=neighbors,
                                # small value: more clusters; large value: less clusters (VERY SENSIBLE)
                                n_components=components,  # small value: more clusters; large value: less clusters
                                metric='cosine').fit_transform(data)
    print("Reduction completed")
    print('\nSaving the UMAP embeddings...')
    f = gzip.GzipFile("umap_embeddings.npy.gz", "w")
    numpy.save(file=f, arr=umap_embeddings)
    f.close()
    print('UMAP embeddings saved')

def k_means_clusters(umap_embeddings, n):
    print("\nClustering...")
    kmeans = KMeans(n_clusters=n, random_state=0).fit(umap_embeddings)
    print("Clustering completed")
    return kmeans


def hdbscan_clusters(min_size, umap_embeddings, eps):

    print("\nClustering...")
    cluster = hdbscan.HDBSCAN(min_cluster_size= min_size,
                              metric='manhattan',
                              cluster_selection_method='eom',
                              prediction_data= True,
                              min_samples=1,
                              cluster_selection_epsilon= eps).fit(umap_embeddings)
    print("Clustering completed")
    return cluster

# Let's implement C-TF-IDF
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

def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])
                     .Doc
                     .count()
                     .reset_index()
                     .rename({"Topic": "Topic", "Doc": "Size"}, axis='columns')
                     .sort_values("Size", ascending=False))
    return topic_sizes


def topic_creation(cluster, comments):
    # Let's create a single document for each cluster
    topics = []
    docs_df = pd.DataFrame(comments, columns=["Doc"])
    docs_df['Topic'] = cluster.labels_
    docs_df['Doc_ID'] = range(len(docs_df))
    docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})
    tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m=len(comments))
    top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)
    #topic_sizes = extract_topic_sizes(docs_df).reset_index(drop=True)

    for i in range(1):
        # Calculate cosine similarity
        similarities = cosine_similarity(tf_idf.T)
        np.fill_diagonal(similarities, 0)

        # Extract label to merge into and from where
        topic_sizes = docs_df.groupby(['Topic']).count().sort_values("Doc", ascending=False).reset_index()
        topic_to_merge = topic_sizes.iloc[-1].Topic
        topic_to_merge_into = np.argmax(similarities[topic_to_merge + 1]) - 1

        # Adjust topics
        docs_df.loc[docs_df.Topic == topic_to_merge, "Topic"] = topic_to_merge_into
        old_topics = docs_df.sort_values("Topic").Topic.unique()
        map_topics = {old_topic: index - 1 for index, old_topic in enumerate(old_topics)}
        docs_df.Topic = docs_df.Topic.map(map_topics)
        docs_per_topic = docs_df.groupby(['Topic'], as_index=False).agg({'Doc': ' '.join})

        # Calculate new topic words
        m = len(comments)
        tf_idf, count = c_tf_idf(docs_per_topic.Doc.values, m)
        top_n_words = extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20)

    topic_sizes = extract_topic_sizes(docs_df).reset_index(drop=True)

    for i in range(len(topic_sizes)):
        topics.append(top_n_words[topic_sizes.at[i, 'Topic']][:20])
        print("\nTopic " + str(i) + "\n", top_n_words[topic_sizes.at[i, 'Topic']][:20])
    with pd.option_context('display.max_rows', 3000):
        print("\nClusters' sizes:\n", topic_sizes)
    return topics

def visualize_clusters(cluster, neighbors, embeddings):
    # We can visualize the clusters
    print("\nPlotting clustering...")
    # Prepare data
    umap_data = umap.UMAP(n_neighbors=neighbors, n_components=2, min_dist=0.0, metric='cosine').fit_transform(embeddings)
    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = cluster.labels_
    # Visualize clusters
    fig, ax = plt.subplots(figsize=(8, 4))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.02)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.02, cmap='hsv_r')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    start_time = time.time()
    path = r'..\reddit_corpus\RC_2015-01'
    emb_path = r'embeddings.npy.gz'
    N = 200000  # number of Reddit comments
    comments, full_comments = get_comments(path, N)
    # save_embeddings(comments)
    embeddings = load_embeddings(emb_path)
    #reduce_dimension(embeddings, 8, 3)
    umap_emb = load_embeddings('umap_embeddings.npy.gz')
    #cluster = hdbscan_clusters(20, umap_emb, 0.4)
    cluster = k_means_clusters(umap_emb, 300)
    #clean_comments = stanza_stem(comments)
    topics = topic_creation(cluster, comments)

    print('\nSaving the topics keywords...')
    f = gzip.GzipFile("topics.npy.gz", "w")
    numpy.save(file=f, arr=topics)
    f.close()
    print('Keywords saved')

    print('\nSaving the cluster model...')
    with open('clusters.pickle', 'wb') as handle:
        pickle.dump(cluster, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Model saved')

    print("\nExecution time: %.2f seconds" % (time.time() - start_time))
    visualize_clusters(cluster, 8, embeddings)
