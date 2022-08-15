import csv
import gzip
from collections import Counter

import numpy as np
from statistics import mean
#from get_all_data import associate_labels, organize_all_comments
#from network import get_topics_popularity
#from sentiment import build_vocabulary
#from sentiment import porter as pt
import spacy, json, re, os
from nltk import RegexpTokenizer
import statistics
from statistics import mean
import emoji
from nltk.stem import PorterStemmer
import collections
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import textstat
from better_profanity import profanity
from tqdm import tqdm
from scipy import stats
import scipy
import scipy.stats as st

def mean_confidence_interval(data, m, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    #se = scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    if m > h:
        print(m-h, m+h)
        return m, (m-h, m+h)
    else:
        print('m-h')
        return m, (0, m+h)


def check_links(comments):
    """
    Filters comments, discarding the ones which contain at least one link
    :param comments: list of comments
    :return: filtered list of comments
    """
    tokenizer = RegexpTokenizer(r'\w+')
    for comment in comments:
        for word in tokenizer.tokenize(comment):
            if word[:4] == 'http':
                comments.remove(comment)
                break
    return comments

def strip_links(comment):
    tokenizer = RegexpTokenizer(r'\w+')
    words = comment.split()
    filtered_words = []
    link_count = 0
    for word in words:
        no_punct = tokenizer.tokenize(word)
        word = ' '.join(no_punct)
        if word[:4] != 'http':
            filtered_words.append(word)
        else:
            link_count += 1
    filtered_comment = ' '.join(filtered_words)
    return filtered_comment, link_count

def links_rate(data, topics):
    links = {}
    for topic in topics:
        n_char = 0
        n_links_tot = 0
        comments = data[topic]
        for comment in comments:
            n_char += get_n_characters(comment)
            _, n_links = strip_links(comment)
            n_links_tot += n_links
        links[topic] = n_links_tot / n_char
    return links


def get_n_sentences(nlp, comment):
    """
    Calculates the number of sentences in a comment
    :param nlp: spacy instance
    :param comment: a single comment
    :return: in order: the list of sentences, the number of sentences
    """
    about_doc = nlp(str(comment))
    sentences = list(about_doc.sents)
    fixed_s = []
    for i, sentence in enumerate(sentences):
        if len(sentence) > 3:
            fixed_s.append(str(sentence))
    return fixed_s, len(fixed_s)

def get_n_characters(comment):
    """
    Calculates the number of characters (only letters and numbers) in a comment
    :param comment: a single comment
    :return: number of characters
    """
    n_letters = sum(c.isalpha() for c in comment)
    n_numbers = sum(c.isdigit() for c in comment)
    n_characters = n_letters + n_numbers

    return n_characters

def get_n_words(comment):
    """
    Computes the number of words in a comment.
    :param comment: a single comment
    :return: in order: the list of words, the number of words
    """
    words = re.split("\\s|'|;|!|,|\\n", comment)
    fixed_words = []
    for word in words:
        if len(word) != 0:
            fixed_words.append(word)
    n_words = len(fixed_words)
    if len(fixed_words) == 0:
        n_words = 1
    return fixed_words, n_words

def ari(comment, nlp):
    """
    Computes the Automated Readability Index (ARI)
    :param comment: a single comment
    :param nlp: a spacy instance
    :return: ARI value for that comment
    """
    _, n_sentences = get_n_sentences(nlp, comment)
    _, n_words = get_n_words(comment)
    n_characters = get_n_characters(comment)
    if n_sentences == 0:
        n_sentences = 1
    formula = 4.71*(n_characters/n_words) + 0.5*(n_words/n_sentences) - 21.43

    return formula

def cli(comment, nlp):
    """
    Computes the Coleman-Liau Index (CLI)
    :param comment: a single comment
    :param nlp: a spacy instance
    :return: CLI value for that comment
    """
    _, n_sentences = get_n_sentences(nlp, comment)
    _, n_words = get_n_words(comment)
    n_characters = get_n_characters(comment)
    if n_sentences == 0:
        n_sentences = 1
    L = (n_characters / n_words) * 100
    S = (n_sentences / n_words) * 100
    formula = 0.0588*L - 0.296*S - 15.8

    return formula

def get_all_indices(comments, topics):
    #nlp = spacy.load('en_core_web_sm')
    scores = {}
    scores_ci = {}

    for topic in topics:
        scores[topic] = []
        scores_ci[topic] = []
        n_reddits = len(comments[topic])

        ari_scores = []
        cli_scores = []
        for i, reddit in enumerate(comments[topic]):
            print('Processing reddits about ' + topic)
            print(str(i) + '/' + str(n_reddits))
            #ari_scores.append(ari(reddit, nlp))
            #cli_scores.append(cli(reddit, nlp))
            ari_scores.append(textstat.automated_readability_index(reddit))
            cli_scores.append(textstat.coleman_liau_index(reddit))

        scores[topic].append(statistics.mean(ari_scores))
        scores[topic].append(statistics.mean(cli_scores))
        
        n = len(ari_scores)

        std_ari = statistics.stdev(ari_scores)
        std_cli = statistics.stdev(cli_scores)

        yerr_ari = std_ari / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)
        yerr_cli = std_cli / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)


        #yerr_ari = st.t.interval(alpha=0.95, df=len(ari_scores)-1, loc=np.mean(ari_scores), scale=st.sem(ari_scores))
        #yerr_cli = st.t.interval(alpha=0.95, df=len(cli_scores)-1, loc=np.mean(cli_scores), scale=st.sem(cli_scores))

        scores_ci[topic].append(yerr_ari)
        scores_ci[topic].append(yerr_cli)

    return scores, scores_ci

def get_reading_time(comments, topics):
    #nlp = spacy.load('en_core_web_sm')
    scores = {}

    for topic in topics:
        scores[topic] = []
        n_reddits = len(comments[topic])

        reading_time = []
        for i, reddit in enumerate(comments[topic]):
            #reading_time.append(textstat.reading_time(reddit, ms_per_char=14.69))
            scores[topic].append(textstat.reading_time(reddit, ms_per_char=14.69))

        #scores[topic].append(statistics.mean(reading_time))

    return scores

def detect_emojis(comments):
    emojis = 0
    words_count = 0
    pop = []
    for comment in comments:
        p_emoji = 0
        words = comment.split()
        words_count+=len(words)
        for word in words:
            
            codes = word.encode('unicode-escape').decode('ASCII').split('\\')
            for code in codes:
                if len(code) > 3:
                    if code[:4] == 'U000':
                        emojis += 1
                        p_emoji += 1
            '''
            if word in emoji.UNICODE_EMOJI and not word == 'it':
                print((comment, word))
                emojis += 1
            '''
        #print(f'{comment} score: {p_emoji} {len(words)} {p_emoji/len(words)}')
        pop.append(p_emoji/len(words))
    #print(pop)
    std = statistics.stdev(pop)
    n = len(pop)
    #yerr = st.t.interval(alpha=0.95, df=len(pop)-1, loc=np.mean(pop), scale=st.sem(pop))
    yerr = mean_confidence_interval(pop, emojis / words_count)
    return emojis / words_count , std / np.sqrt(n) * stats.t.ppf(1-0.10/2, n - 1)

def detect_keyboard_emojis(comments):
    emojis_list = [':‑(', ':(', ':‑c', ':c', ':‑<', ':<', ':‑[', ':[', ':-||', '>:[', ':{', ':@',
         ':(', ';(', ':‑)', ':)', ':-]', ':]', ':-3', ':3', ':->', ':>', '8-)', '8)',
         ':-}', ':}', ':o)', ':c)', ':^)', '=]', '=)', ':‑D', ':D', '8‑D', '8D', 'x‑D',
         'xD', 'X‑D', 'XD', '=D', '=3', 'B^D', 'c:', 'C:', ':-))', ":'‑(", ":'(", ":'‑)",
         ":')", "D‑':", 'D:<', 'D:', 'D8', 'D;', 'D=', 'DX', ':‑O', ':O', ':‑o', ':o',
         ':-0', '8‑0', '>:O', ':-*', ':*', ':×', ';‑)', ';)', '*-)', '*)', ';‑]', ';]',
         ';^)', ';>', ':‑,', ';D', ':‑P', ':P', 'X‑P', 'XP', 'x‑p', 'xp', ':‑p', ':p',
         ':‑Þ', ':Þ', ':‑þ', ':þ', ':‑b', ':b', 'd:', '=p', '>:P', ':-/', ':/', ':‑.',
         '>:\\', '>:/', ':\\', '=/', '=\\', ':L', '=L', ':S', ':‑|', ':|', '<_<', '>_>',
         '</3', '<\\3', '<3']
    emojis = 0
    words_count = 0
    pop = []
    for comment in comments:
        p_emoji = 0
        words = comment.split()
        words_count+=len(words)
        for word in words:
            if word in emojis_list:
                emojis += 1
                p_emoji += 1
        pop.append(p_emoji/len(words))
        #print(f'{comment} score: {p_emoji} {len(words)} {p_emoji/len(words)}')
    
    std = statistics.stdev(pop)
    n = len(pop)
    #yerr = st.t.interval(alpha=0.95, df=len(pop)-1, loc=np.mean(pop), scale=st.sem(pop))
    yerr = mean_confidence_interval(pop, emojis / words_count)
    #return yerr[0], yerr[1]
    #return emojis / words_count, yerr
    return emojis / words_count , std / np.sqrt(n) * stats.t.ppf(1-0.10/2, n - 1)

def get_all_emojis(data, topics):
    """
    Computes the average number of emojis for each comment (divided by topic)
    :param data: dictionary with the following structure:
                     - key:   topic name
                     - value: list of comments associated to that topic
    :return: dictionary with the following structure:
                     - key:   topic name
                     - value: average number of emojis
    """
    emoji_counts = {}
    emoji_yerr = {}
    k_emoji_counts = {}
    k_emoji_yerr = {}
    for topic in topics:
        comments = data[topic]
        emoji_counts[topic], emoji_yerr[topic] = detect_emojis(comments)
        k_emoji_counts[topic], k_emoji_yerr[topic] = detect_keyboard_emojis(comments)
    return emoji_counts, k_emoji_counts, emoji_yerr, k_emoji_yerr

def count_bad_words(texts):
    bad_words_count = 0
    n_words = 0
    pop = []
    for text, _ in zip(texts, tqdm(range(len(texts)))):
        p_bad= 0
        censored_text = profanity.censor(text)
        n_words += len(censored_text.split(' '))
        for word in censored_text.split(' '):
            if word == "****":
                bad_words_count+=1
                p_bad+=1
                #print(f"{text} {p_bad/len(censored_text.split(' '))}")
        pop.append(p_bad/len(censored_text.split(' ')))

    std = statistics.stdev(pop)
    n = len(pop)
    #yerr = st.t.interval(alpha=0.95, df=len(pop)-1, loc=np.mean(pop), scale=st.sem(pop))
    yerr = std / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)
    return n_words, bad_words_count, yerr, pop

def swearing_count(data, topics):
    """
    Computes the percentage of swearing words over the total number of words, for each topic
    :param data: dictionary with the following structure:
                     - key:   topic name
                     - value: list of comments associated to that topic
    :return:
    """
    with open(r'facebook-bad-words-list_comma-separated-text-file_2021_01_18.txt', encoding='utf8') as f:
        for i, line in enumerate(f):
            if i == 14:
                _ = line.split(',')
    swearings = []
    for swearing in _:
        swearings.append(swearing.strip())
    print(swearings)
    swear_rate = {}
    swear_rate_ci = {}
    tokenizer = RegexpTokenizer(r'\w+')
    for topic in topics:
        swear_count = 0
        n_words = 0
        comments = data[topic]
        '''
        for comment in comments:
            _, n = get_n_words(comment)
            n_words += n
            comment_no_punct = tokenizer.tokenize(comment)
            lower_comment = [w.lower() for w in comment_no_punct]
            for word in lower_comment:
                if word in swearings:
                    swear_count += 1
        '''
        n_words, swear_count, yerr, pop = count_bad_words(comments)
        swear_rate[topic] = swear_count / n_words
        swear_rate_ci[topic] = yerr
        #swear_rate[topic], swear_rate_ci[topic] = mean_confidence_interval(pop,swear_count / n_words)
    return swear_rate, swear_rate_ci

def get_words_stop(data):
    """
    Reads the text passed as input, formatting the words in a more suitable structure (no punctuation,
    lower case); it is also applied the stopwords technique.
    :param data: list of texts
    :return: list of words in the text
    """
    #nlp = StanfordCoreNLP(r'/home/marco/projects/reddit_twitter_analysis/CoreNlp/stanford-corenlp-full-2018-10-05')
    
    #props={'annotators': 'tokenize,ssplit,pos,lemma','pipelineLanguage':'en','outputFormat':'json'}
    stemmer = PorterStemmer()
    words = []
    p = "|\"#%$()'+,-./;:<=>?`!@[]^_{}-"
    table = str.maketrans(p, " " * len(p))
    for comment in data:
        
        formatted_line = comment.translate(table).lower()
        for w in formatted_line.split():
            if len(w) > 2 and w.lower() not in stopwords.words('english'):
                words.append(stemmer.stem(w)) #stemming
        '''
        r = nlp.annotate(comment, properties=props)
        for sentence in json.loads(r)['sentences']:
            for token in sentence['tokens']:
                if not token['lemma'] in stopwords.words('english'):
                    words.append(token['lemma'])
        '''
    return words

def create_voc(data):
    """
    Uses the 'get_words' functions to format the text and builds the structure of the vocabulary
    :param train_file: file used to build the vocabulary
    :param stop: flag to enable stopping technique
    :param stem: flag to enable stemming technique
    :return: two vocabularies, one about the titles, the other about the publishers' names
    """
    voc = collections.Counter()
    words = get_words_stop(data)
    k = (len(words)/100) * 50
    print(f'Count {len(words)}')
    voc.update(words)
    return voc, int(k)

def write_voc(voc, filename, n):
    """
    Saves the vocabulary as a txt file, keeping the n most frequent words
    :param voc: vocabulary to save
    :param filename: name to be assigned to the new file
    :param n: number of most frequent words to keep
    """
    f = open(filename, "w", encoding="utf8")
    print(voc.most_common(10))
    for word, count in sorted(voc.most_common(n)):
        print(word, file=f)
    f.close()

def oov(data, n, topics):
    """
    Computes the percentage of words that are not in the top used words in a set of comments
    (for a certain topic)
    :param data: dictionary with the following structure:
                     - key:   topic name
                     - value: list of comments associated to that topic
    :param n: size of the top words
    :return: dictionary with the following structure:
                 - key:   topic name
                 - value: oov rate
    """
    tokenizer = RegexpTokenizer(r'\w+')
    stemmer = PorterStemmer()
    voc_name = 'topic_voc.txt'
    oov_dict = {}
    oov_dict_ci = {}
    pop = []
    for topic in topics:
        voc = []
        n_words = 0
        n_oov = 0
        comments = data[topic]
        voc, n = create_voc(data[topic])
        write_voc(voc, voc_name, n)
        voc = []
        with open(voc_name, encoding='utf8') as f:
            for i, line in enumerate(f):
                voc.append(line.strip())
        for comment in comments:
            p_ovv = 0
            no_punct = tokenizer.tokenize(comment)
            stem = [stemmer.stem(w.lower()) for w in no_punct]
            n_words += len(stem)
            for word in stem:
                if word not in voc:
                    n_oov += 1
                    p_ovv += 1
            if len(stem) > 0:
                if p_ovv/len(stem) < 1:
                    pop.append(p_ovv/len(stem))
                else:
                    pop.append(1)
        std = statistics.stdev(pop)
        n = len(pop)
        yerr = std / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)
        oov_dict[topic] = n_oov / n_words
        #oov_dict_ci[topic] = st.t.interval(alpha=0.95, df=len(pop)-1, loc=np.mean(pop), scale=st.sem(pop))
        oov_dict_ci[topic] = yerr
        #oov_dict[topic], oov_dict_ci[topic] = mean_confidence_interval(pop,n_oov / n_words)
    return oov_dict, oov_dict_ci

def uppercase_rate(data, topics):
    """
    Computes number of uppercase characters over the total number of characters
    (excluded punctuation, blank spaces, links and numbers)
    :param data: dictionary with the following structure:
                     - key:   topic name
                     - value: list of comments associated to that topic
    :return: dictionary with the following structure:
             - key:  topic
             - value: uppercase rate
    """
    upper_rate = {}
    for topic in topics:
        tot_upper = 0
        tot_char = 0
        comments = data[topic]
        for comment in comments:
            filtered_comment = strip_links(comment)
            #print(filtered_comment)
            tot_upper += sum(map(str.isupper, filtered_comment[0]))
            tot_char += get_n_characters(filtered_comment[0])
            #if tot_upper > 5:
                #print(f'{filtered_comment} {sum(map(str.isupper, filtered_comment[0]))/get_n_characters(comment)}')
        upper_rate[topic] = tot_upper / tot_char

    return upper_rate

def uppercase_words(data, topics):
    """
    Computes the percentage of words in uppercase over the total number of words, for each
    topic
    :param data: dictionary with the following structure:
                     - key:   topic name
                     - value: list of comments associated to that topic
    :return: dictionary with the following structure:
             - key:  topic
             - value: uppercase rate
    """
    upper = {}
    upper_ci = {}
    pop = []
    for topic in topics:
        upper_words = 0
        tot_words = 0
        comments = data[topic]
        for comment in comments:
            _, n = get_n_words(comment)
            p_upper = 0
            tot_words += n
            words = comment.split()
            for word in words:
                if word.isupper():
                    upper_words += 1
                    p_upper += 1
                    #print(f'{comment} {p_upper/n}')
            pop.append(p_upper/n)

        std = statistics.stdev(pop)
        n = len(pop)
        #yerr = std / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)
        upper[topic] = upper_words / tot_words
        #upper_ci[topic] = st.t.interval(alpha=0.95, df=len(pop)-1, loc=np.mean(pop), scale=st.sem(pop))
        #upper_ci[topic] = mean_confidence_interval(pop)
        print(pop / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1))
        upper_ci[topic] = std / np.sqrt(n) * stats.t.ppf(1-0.05/2, n - 1)
        #upper[topic], upper_ci[topic] = mean_confidence_interval(pop, upper_words / tot_words)
    return upper, upper_ci

def get_similarity_metrics(p):
    """
    Reads the Pederson's metric about similarity between the topic and its top words
    :param p: path to the files up to the 'reddit' or 'twitter' directory (included)
    :return: dictionary with the following structure:
                 - key: topic name
                 - value: list of the 8 means of the top words' similarity metrics
    """
    path_reddit = p
    reddit_values = {}
    topics = os.listdir(path_reddit)
    for topic in topics:
        reddit_values[topic] = []
        wup = []
        jcn = []
        lhc = []
        lin = []
        res = []
        pth = []
        lesk = []
        hso = []
        path_topic = path_reddit + r'\\' + topic
        files = os.listdir(path_topic)
        for file in files:
            path_file = path_topic + r'\\' + file
            print('\n')
            with open(path_file) as f:
                for i, line in enumerate(f.readlines()):
                    if i == 0:
                        wup.append(float(line.split()[6]))
                    if i == 1:
                        jcn.append(float(line.split()[6]))
                    if i == 2:
                        lhc.append(float(line.split()[6]))
                    if i == 3:
                        lin.append(float(line.split()[6]))
                    if i == 4:
                        res.append(float(line.split()[6]))
                    if i == 5:
                        pth.append(float(line.split()[6]))
                    if i == 6:
                        lesk.append(float(line.split()[6]))
                    if i == 7:
                        hso.append(float(line.split()[6]))
        reddit_values[topic].append(mean(wup))
        reddit_values[topic].append(mean(jcn))
        reddit_values[topic].append(mean(lhc))
        reddit_values[topic].append(mean(lin))
        reddit_values[topic].append(mean(res))
        reddit_values[topic].append(mean(pth))
        reddit_values[topic].append(mean(lesk))
        reddit_values[topic].append(mean(hso))
    return reddit_values

def get_emojis_all_trimesters(topic_paths, clusters, comments_paths, trimesters):
    """
    Computes the mean number of emojis used for each comment, realizing a cross-topic metric
    They are analized a number of trimesters according to the ones passed by the users
    :param topic_paths: list of paths to the 'topic.txt' files
    :param clusters: list of paths to the cluster objects
    :param comments_paths: list of paths to the raw comments
    :param trimesters: list of names to identify the different trimesters
    :return: dictionary with the following structure:
                 - key: trimester
                 - value: mean number (cross-topic) of emojis used in that trimester
    """
    mean_emojis = {}
    mean_kemojis = {}
    for i, tp in enumerate(topic_paths):
        print('\nReading comments: ', trimesters[i])
        cluster_path = clusters[i]
        path1, path2, path3 = comments_paths[i]
        f1 = gzip.GzipFile(path1, "r")
        f2 = gzip.GzipFile(path2, "r")
        f3 = gzip.GzipFile(path3, "r")
        comments1 = np.load(f1)[:100000]
        comments2 = np.load(f2)[:100000]
        comments3 = np.load(f3)[:100000]
        comments = np.concatenate((comments1, comments2, comments3), axis=None)
        print('Processing comments: ', trimesters[i])
        labels = associate_labels(tp)
        organized_comm = organize_all_comments(cluster_path, comments, labels)
        emojis, k_emojis = get_all_emojis(organized_comm)
        topics = emojis.keys()
        emoji_values = []
        kemoji_values = []
        for topic in topics:
            emoji_values.append(emojis[topic] / len(organized_comm[topic]))
            kemoji_values.append(k_emojis[topic] / len(organized_comm[topic]))
        mean_emojis[trimesters[i]] = mean(emoji_values)
        mean_kemojis[trimesters[i]] = mean(kemoji_values)

    return mean_emojis, mean_kemojis

def get_swearings_all_trimesters(topic_paths, clusters, comments_paths, trimesters):
    mean_swearings = {}
    for i, tp in enumerate(topic_paths):
        print('\nReading comments: ', trimesters[i])
        cluster_path = clusters[i]
        path1, path2, path3 = comments_paths[i]
        f1 = gzip.GzipFile(path1, "r")
        f2 = gzip.GzipFile(path2, "r")
        f3 = gzip.GzipFile(path3, "r")
        comments1 = np.load(f1)[:100000]
        comments2 = np.load(f2)[:100000]
        comments3 = np.load(f3)[:100000]
        comments = np.concatenate((comments1, comments2, comments3), axis=None)
        print('Processing comments: ', trimesters[i])
        labels = associate_labels(tp)
        organized_comm = organize_all_comments(cluster_path, comments, labels)
        trimester_swear = swearing_count(organized_comm)
        topics = trimester_swear.keys()
        swearing_values = []
        for topic in topics:
            swearing_values.append(trimester_swear[topic])
        mean_swearings[trimesters[i]] = mean(swearing_values)
    return mean_swearings

def get_readability_all_trimesters(topic_paths, clusters, comments_paths, trimesters):
    mean_cli = {}
    mean_ari = {}
    for i, tp in enumerate(topic_paths):
        print('\nReading comments: ', trimesters[i])
        cluster_path = clusters[i]
        path1, path2, path3 = comments_paths[i]
        f1 = gzip.GzipFile(path1, "r")
        f2 = gzip.GzipFile(path2, "r")
        f3 = gzip.GzipFile(path3, "r")
        comments1 = np.load(f1)[:100000]
        comments2 = np.load(f2)[:100000]
        comments3 = np.load(f3)[:100000]
        comments = np.concatenate((comments1, comments2, comments3), axis=None)
        print('Processing comments: ', trimesters[i])
        labels = associate_labels(tp)
        organized_comm = organize_all_comments(cluster_path, comments, labels)
        trimester_read = get_all_indices(organized_comm)
        topics = trimester_read.keys()
        ari_values = []
        cli_values = []
        for topic in topics:
            ari_values.append(trimester_read[topic][0])
            cli_values.append(trimester_read[topic][1])
        mean_ari[trimesters[i]] = mean(ari_values)
        mean_cli[trimesters[i]] = mean(cli_values)
    return mean_ari, mean_cli


def get_trimesters_specificity(topic_paths, clusters, comments_paths, trimesters):
    all_topics = []
    trimesters_topics = {}
    trimesters_specificity = {}
    for i, trimester in enumerate(trimesters):
        trimesters_topics[trimester] = []
        path1, path2, path3 = comments_paths[i]
        popularities = get_topics_popularity(topic_paths[i], path1, path2, path3, clusters[i])
        for j, topic in enumerate(popularities.keys()):
            trimesters_topics[trimester].append(topic)
            all_topics.append(topic)
            if j == 6:
                break
    topics_freq = Counter(all_topics)
    for trimester in trimesters:
        trimester_specificity = []
        for topic in trimesters_topics[trimester]:
            trimester_specificity.append(len(all_topics) / topics_freq[topic])
        trimesters_specificity[trimester] = mean(trimester_specificity)
    return trimesters_specificity

def get_all_topics_adhesions(topic_paths, clusters, comments_paths, trimesters):
    all_topics = []
    trimesters_topics = {}
    for i, trimester in enumerate(trimesters):
        trimesters_topics[trimester] = []
        path1, path2, path3 = comments_paths[i]
        popularities = get_topics_popularity(topic_paths[i], path1, path2, path3, clusters[i])
        for i, topic in enumerate(popularities.keys()):
            trimesters_topics[trimester].append(topic)
            all_topics.append(topic)
            if i == 9:
                break
    adhesion = len(all_topics) / len(list(set(all_topics)))
    return adhesion

def get_trimesters_adhesions(topic_paths, clusters, comments_paths, trimesters):
    all_topics = []
    trimesters_topics = {}
    trimesters_adhesions = {}
    for i, trimester in enumerate(trimesters):
        trimesters_topics[trimester] = []
        path1, path2, path3 = comments_paths[i]
        popularities = get_topics_popularity(topic_paths[i], path1, path2, path3, clusters[i])
        for j, topic in enumerate(popularities.keys()):
            trimesters_topics[trimester].append(topic)
            all_topics.append(topic)
            if j == 9:
                break
        if i != 0:
            differences = len(set(trimesters_topics[trimesters[i]]) - set(trimesters_topics[trimesters[i-1]]))
            trimesters_adhesions[trimester] = (differences / len(trimesters_topics[trimester])) * 100
    return trimesters_adhesions


if __name__ == '__main__':
    trimesters = ['2015_1st', '2015_2nd']

    r_topic_paths = [r'C:\Users\zippo\Desktop\results\reddit\2015\1st_trim\topics.txt',
                     r'C:\Users\zippo\Desktop\results\reddit\2015\2nd_trim\topics.txt']

    r_comments_paths = [[r"data\reddit\2015\1st_trim\reddit_2015_01",
                         r"data\reddit\2015\1st_trim\reddit_2015_02",
                         r"data\reddit\2015\1st_trim\reddit_2015_03"],
                        [r"data\reddit\2015\2nd_trim\reddit_2015_09",
                         r"data\reddit\2015\2nd_trim\reddit_2015_10",
                         r"data\reddit\2015\2nd_trim\reddit_2015_11"]
                        ]

    r_clusters = ['clustersR2015_1st.pickle', 'clustersR2015_2nd.pickle']

    x = get_trimesters_adhesions(r_topic_paths, r_clusters, r_comments_paths, trimesters)

    #print(x)






