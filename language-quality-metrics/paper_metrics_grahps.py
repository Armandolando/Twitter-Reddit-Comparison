import gzip
import matplotlib.pyplot as plt
import json
import numpy as np
#from get_all_data import associate_labels, organize_all_comments
#from network import get_topics_popularity
from paper_metrics import get_all_emojis, swearing_count, oov, uppercase_rate, uppercase_words, \
    links_rate, get_similarity_metrics, get_emojis_all_trimesters, get_swearings_all_trimesters, \
    get_readability_all_trimesters, get_trimesters_specificity, get_trimesters_adhesions, get_all_topics_adhesions, get_all_indices, get_reading_time
import pandas as pd
from statistics import mean
import os
from tqdm import trange
import scipy.stats as st
import itertools

def limit_zero(yerr, mean):
    upper = []
    lower = []
    for y, m in zip(yerr, mean):
        upper.append(y)
        if m-y > 0:
            lower.append(y)
        else:
            lower.append(y - (y- m))
    return [lower, upper]

def print_confidence_interval(data):
    print(st.t.interval(alpha=0.95, df=len(data)-1, loc=np.mean(data), scale=st.sem(data)))

def plot_read_indices(reddit_scores, twitter_scores, topics, topics_lables):
    reddit_scores, reddit_scores_ci = get_all_indices(reddit_scores, topics)
    twitter_scores, twitter_scores_ci = get_all_indices(twitter_scores, topics)
    reddit_ari = []
    #reddit_ari_ci = [[], []]
    reddit_ari_ci = []
    reddit_cli = []
    reddit_cli_ci = [[], []]
    reddit_cli_ci = []
    twitter_ari = []
    twitter_ari_ci = [[], []]
    twitter_ari_ci = []
    twitter_cli = []
    twitter_cli_ci = [[], []]
    twitter_cli_ci = []
    for topic in topics:
        reddit_ari.append(reddit_scores[topic][0])
        reddit_ari_ci.append(reddit_scores_ci[topic][0])
        #reddit_ari_ci[0].append(reddit_scores_ci[topic][0][0])
        #reddit_ari_ci[1].append(reddit_scores_ci[topic][0][1])
        reddit_cli.append(reddit_scores[topic][1])
        reddit_cli_ci.append(reddit_scores_ci[topic][1])
        #reddit_cli_ci[0].append(reddit_scores_ci[topic][1][0])
        #reddit_cli_ci[1].append(reddit_scores_ci[topic][1][1])
        twitter_ari.append(twitter_scores[topic][0])
        twitter_ari_ci.append(twitter_scores_ci[topic][0])
        #twitter_ari_ci[0].append(twitter_scores_ci[topic][0][0])
        #twitter_ari_ci[1].append(twitter_scores_ci[topic][0][1])
        twitter_cli.append(twitter_scores[topic][1])
        twitter_cli_ci.append(twitter_scores_ci[topic][1])
        #twitter_cli_ci[0].append(twitter_scores_ci[topic][1][0])
        #twitter_cli_ci[1].append(twitter_scores_ci[topic][1][1])

    print_confidence_interval(reddit_ari+twitter_ari)
    print_confidence_interval(reddit_cli+twitter_cli)
    # Set position of bar on X axis
    ind = np.arange(len(reddit_scores.keys()))  # the x locations for the groups
    width = 0.18
    edge_width = 0.75
    bar1 = plt.bar(ind, reddit_ari, width, yerr=reddit_ari_ci, ecolor='black', capsize=7, color='orange', label='Reddit scores', edgecolor='black', linewidth=edge_width)
    bar2 = plt.bar(ind + width, reddit_cli, width, yerr=reddit_cli_ci, ecolor='black', capsize=7, color='orange', edgecolor='black', linewidth=edge_width)
    bar3 = plt.bar(ind + 2*width, twitter_ari, width, yerr=twitter_ari_ci, capsize=7, ecolor='black', color='blue', label='Twitter scores', edgecolor='black', linewidth=edge_width)
    bar4 = plt.bar(ind + 3*width, twitter_cli, width, yerr=twitter_cli_ci, capsize=7, ecolor='black', color='blue', edgecolor='black', linewidth=edge_width)


    for bar in bar1:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'ARI', ha='center', va='bottom', fontweight='bold', fontsize=13, color='blue', rotation=90)
    for bar in bar2:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'CLI', ha='center', va='bottom', fontweight='bold', fontsize=13, color='blue', rotation=90)
    for bar in bar3:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'ARI', ha='center', va='bottom', fontweight='bold', fontsize=13, color='orange', rotation=90)
    for bar in bar4:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2.0, 0, 'CLI', ha='center', va='bottom', fontweight='bold', fontsize=13, color='orange', rotation=90)

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    plt.ylabel('ARI and CLI scores', fontweight='bold', fontsize=20)
    plt.xticks([r + width for r in range(len(topics_lables))], topics_lables, rotation=25, fontsize=20)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_emojis(reddit, twitter, topics, topics_lables, K):
    reddit_emojis, reddit_k_emojis, reddit_ci, reddit_k_ci = get_all_emojis(reddit, topics)
    twitter_emojis, twitter_k_emojis, twitter_ci, twitter_k_ci = get_all_emojis(twitter, topics)
    r_emojis_bars = []
    #r_emojis_bars_ci = [[], []]
    r_emojis_bars_ci = []
    r_k_emojis_bars = []
    #r_k_emojis_bars_ci = [[], []]
    r_k_emojis_bars_ci = []
    t_emojis_bars = []
    #t_emojis_bars_ci = [[], []]
    t_emojis_bars_ci = []
    t_k_emojis_bars = []
    #t_k_emojis_bars_ci = [[], []]
    t_k_emojis_bars_ci = []
    print(reddit_ci)
    for topic in topics:
        '''
        r_emojis_bars.append(reddit_emojis[topic] / len(reddit[topic]))
        r_k_emojis_bars.append(reddit_k_emojis[topic] / len(reddit[topic]))
        t_emojis_bars.append(twitter_emojis[topic] / len(twitter[topic]))
        t_k_emojis_bars.append(twitter_k_emojis[topic] / len(twitter[topic]))
        '''
        #print(topic, reddit_ci[topic][0], reddit_ci[topic][1])
        r_emojis_bars.append(reddit_emojis[topic])
        r_emojis_bars_ci.append(reddit_ci[topic])
        
        #r_emojis_bars_ci[0].append(reddit_ci[topic][0])
        #r_emojis_bars_ci[1].append(reddit_ci[topic][1])
        r_k_emojis_bars.append(reddit_k_emojis[topic])
        r_k_emojis_bars_ci.append(reddit_k_ci[topic])
        
        #r_k_emojis_bars_ci[0].append(reddit_k_ci[topic][0])
        #r_k_emojis_bars_ci[1].append(reddit_k_ci[topic][1])
        t_emojis_bars.append(twitter_emojis[topic])
        t_emojis_bars_ci.append(twitter_ci[topic])
        
        #t_emojis_bars_ci[0].append(twitter_ci[topic][0])
        #t_emojis_bars_ci[1].append(twitter_ci[topic][1])
        t_k_emojis_bars.append(twitter_k_emojis[topic])
        t_k_emojis_bars_ci.append(twitter_k_ci[topic])
        
        #t_k_emojis_bars_ci[0].append(twitter_k_ci[topic][0])
        #t_k_emojis_bars_ci[1].append(twitter_k_ci[topic][1])
        # Set position of bar on X axis

    r_emojis_bars_ci = limit_zero(r_emojis_bars_ci, r_emojis_bars)
    r_k_emojis_bars_ci = limit_zero(r_k_emojis_bars_ci, r_k_emojis_bars)
    t_emojis_bars_ci = limit_zero(t_emojis_bars_ci, t_emojis_bars)
    t_k_emojis_bars_ci = limit_zero(t_k_emojis_bars_ci, t_k_emojis_bars)

    print(r_emojis_bars_ci)

    print_confidence_interval(r_emojis_bars+t_emojis_bars)
    print_confidence_interval(r_k_emojis_bars+t_k_emojis_bars)

    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    if K:
        plt.bar(ind, r_k_emojis_bars, width, yerr=r_k_emojis_bars_ci, ecolor='black', capsize=7, color='orange', label='Reddit Emoticons Rate')
        plt.bar(ind + width, t_k_emojis_bars, width, yerr=t_k_emojis_bars_ci, ecolor='black', capsize=7, color='blue', label='Twitter Emoticons Rate')
        plt.ylabel('Emoticons Rate', fontweight='bold', fontsize=20)
    else:
        plt.bar(ind, r_emojis_bars, width, yerr=r_emojis_bars_ci, ecolor='black', capsize=7, color='orange', label='Reddit Emojis Rate')
        plt.bar(ind + width, t_emojis_bars, width, yerr=t_emojis_bars_ci, ecolor='black', capsize=7, color='blue', label='Twitter Emojis Rate')
        plt.ylabel('Emojis Rate', fontweight='bold', fontsize=20)

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    
    plt.xticks([r + width for r in range(len(topics_lables))], topics_lables, fontsize=20, rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_reading_time(reddits, tweets, topics, topics_lables):
    reddit_swearings = get_reading_time(reddits, topics)
    twitter_swearings = get_reading_time(tweets, topics)
    reddit_bars = []
    twitter_bars = []
    #print(reddit_swearings)
    for topic in topics:
        reddit_bars.append(reddit_swearings[topic])
        twitter_bars.append(twitter_swearings[topic])

    print_confidence_interval(list(itertools.chain.from_iterable(reddit_bars))+list(itertools.chain.from_iterable(twitter_bars)))

    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2

    fig, (ax1, ax2) = plt.subplots(2 ,1)

    ax1.boxplot(reddit_bars, showfliers=False)
    ax1.set_title('Reddit',fontsize=20)
    ax2.boxplot(twitter_bars, showfliers=False)
    ax2.set_title('Twitter',fontsize=20)
    
    # Add xticks on the middle of the group bars
    ax1.set_xlabel('Topics', fontweight='bold')
    ax1.set_ylabel('Reading Time in seconds', fontweight='bold',fontsize=20)
    ax1.set_xticklabels(topics_lables,fontsize=20, rotation=25)

    ax2.set_xlabel('Topics', fontweight='bold',fontsize=20)
    ax2.set_ylabel('Reading Time in seconds', fontweight='bold',fontsize=20)
    ax2.set_xticklabels(topics_lables,fontsize=20, rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)

    # Create legend & Show graphic
    plt.grid()
    plt.show()

def plot_swearings(reddits, tweets, topics, topics_lables):
    reddit_swearings, reddit_swearings_ci = swearing_count(reddits, topics)
    twitter_swearings, twitter_swearings_ci = swearing_count(tweets, topics)
    reddit_bars = []
    #reddit_bars_ci = [[], []]
    reddit_bars_ci = []
    twitter_bars = []
    #twitter_bars_ci = [[], []]
    twitter_bars_ci = []
    for topic in topics:
        reddit_bars.append(reddit_swearings[topic])
        reddit_bars_ci.append(reddit_swearings_ci[topic])
        #reddit_bars_ci[0].append(reddit_swearings_ci[topic][0])
        #reddit_bars_ci[1].append(reddit_swearings_ci[topic][1])
        twitter_bars.append(twitter_swearings[topic])
        twitter_bars_ci.append(twitter_swearings_ci[topic])
        #twitter_bars_ci[0].append(twitter_swearings_ci[topic][0])
        #twitter_bars_ci[1].append(twitter_swearings_ci[topic][1])

    reddit_bars_ci = limit_zero(reddit_bars_ci, reddit_bars)
    twitter_bars_ci = limit_zero(twitter_bars_ci, twitter_bars)

    print_confidence_interval(reddit_bars+twitter_bars)
    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, reddit_bars, width, yerr=reddit_bars_ci, capsize=7, ecolor='black', color='orange', label='Reddit Swearings Rate')
    plt.bar(ind + width, twitter_bars, width, yerr=twitter_bars_ci, capsize=7,  ecolor='black', color='blue', label='Twitter Swearings Rate')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    plt.ylabel('Swearing Word Rate', fontweight='bold', fontsize=20)
    plt.xticks([r + width for r in range(len(topics_lables))], topics_lables, rotation=25, fontsize=20)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_oov_bars(reddits, tweets, topics, topics_lables):
    n = 5000
    reddit_oov, reddit_oov_ci = oov(reddits, n, topics)
    twitter_oov, twitter_oov_ci = oov(tweets, n, topics)
    reddit_bars = []
    reddit_bars_ci = [[], []]
    reddit_bars_ci = []
    twitter_bars = []
    twitter_bars_ci = [[], []]
    twitter_bars_ci = []
    for topic in topics:
        reddit_bars.append(reddit_oov[topic])
        reddit_bars_ci.append(reddit_oov_ci[topic])
        #reddit_bars_ci[0].append(reddit_oov_ci[topic][0])
        #reddit_bars_ci[1].append(reddit_oov_ci[topic][1])
        twitter_bars.append(twitter_oov[topic])
        twitter_bars_ci.append(twitter_oov_ci[topic])
        #twitter_bars_ci[0].append(twitter_oov_ci[topic][0])
        #twitter_bars_ci[1].append(twitter_oov_ci[topic][1])
    print_confidence_interval(reddit_bars+twitter_bars)
    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, reddit_bars, width, yerr=reddit_bars_ci, capsize=7, ecolor='black', color='orange', label='Reddit')
    plt.bar(ind + width, twitter_bars, width, yerr=twitter_bars_ci, capsize=7, ecolor='black', color='blue', label='Twitter')
    plt.ylim([0,1])
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold',fontsize=20)
    plt.ylabel('OOV Rate', fontweight='bold',fontsize=20)
    plt.xticks([r + width for r in range(len(topics_lables))], topics_lables,fontsize=20,rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_oov_graph(reddits, tweets, start, stop, step, topics):
    voc_n = np.arange(start, stop, step)
    reddit_bars = []
    twitter_bars = []
    for n in voc_n:
        reddit_oov = oov(reddits, n, topics)
        twitter_oov = oov(tweets, n, topics)
        reddit_values = []
        twitter_values = []
        for topic in topics:
            reddit_values.append(reddit_oov[topic])
            twitter_values.append(twitter_oov[topic])
        reddit_bars.append(mean(reddit_values))
        twitter_bars.append(mean(twitter_values))

    plt.plot(voc_n, reddit_bars, color='orange', label='Reddit')
    plt.plot(voc_n, twitter_bars, color='blue', label='Twitter')
    plt.ylim([0, 1])

    # Add xticks on the middle of the group bars
    plt.xlabel('Vocabulary Dimension', fontweight='bold')
    plt.ylabel('OOV Rate', fontweight='bold')

    # Create legend & Show graphic
    plt.title('OOV rate with vocabulary dimension from ' + str(start) + ' to ' + str(stop-1))
    plt.grid()
    plt.legend()
    plt.show()

def plot_uppercase_char_rate(reddits, tweets, topics):
    reddit_rate = uppercase_rate(reddits, topics)
    twitter_rate = uppercase_rate(tweets, topics)
    reddit_bars = []
    twitter_bars = []
    for topic in topics:
        reddit_bars.append(reddit_rate[topic])
        twitter_bars.append(twitter_rate[topic])

    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.ylabel('Uppercase Rate', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('Uppercase Rate')
    plt.grid()
    plt.legend()
    plt.show()

def plot_uppercase_word_rate(reddits, tweets, topics, topics_lables):
    reddit_rate, reddit_rate_ci = uppercase_words(reddits, topics)
    twitter_rate, twitter_rate_ci = uppercase_words(tweets, topics)
    reddit_bars = []
    #reddit_bars_ci = [[], []]
    reddit_bars_ci = []
    twitter_bars = []
    #twitter_bars_ci = [[], []]
    twitter_bars_ci = []
    for topic in topics:
        #print(topic, reddit_rate_ci[topic][0], reddit_rate_ci[topic][1])
        reddit_bars.append(reddit_rate[topic])
        reddit_bars_ci.append(reddit_rate_ci[topic])
        #reddit_bars_ci[0].append(reddit_rate_ci[topic][0])
        #reddit_bars_ci[1].append(reddit_rate_ci[topic][1])
        twitter_bars.append(twitter_rate[topic])
        twitter_bars_ci.append(twitter_rate_ci[topic])
        #twitter_bars_ci[0].append(twitter_rate_ci[topic][0])
        #twitter_bars_ci[1].append(twitter_rate_ci[topic][1])
    print_confidence_interval(reddit_bars+twitter_bars)
    print(reddit_bars_ci)
    print(twitter_bars_ci)
    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, reddit_bars, width, yerr=reddit_bars_ci, capsize=7, ecolor='black', color='orange', label='Reddit')
    plt.bar(ind + width, twitter_bars, width, yerr=twitter_bars_ci, capsize=7, ecolor='black', color='blue', label='Twitter')
    
    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold', fontsize=20)
    plt.ylabel('Uppercase Word Rate', fontweight='bold', fontsize=20)
    plt.xticks([r + width for r in range(len(topics_lables))], topics_lables, fontsize=20, rotation=25)
    plt.rcParams.update({'font.size': 20})
    plt.xticks(fontsize=20)
    # Create legend & Show graphic
    plt.grid()
    plt.legend()
    plt.show()

def plot_similarity(reddit_data, twitter_data):
    topics = reddit_data.keys()
    metrics = ['wup', 'jcn', 'lhc', 'lin', 'res', 'path', 'lesk', 'hso']
    wup_reddit_bars = []
    wup_twitter_bars = []
    jcn_reddit_bars = []
    jcn_twitter_bars = []
    lhc_reddit_bars = []
    lhc_twitter_bars = []
    lin_reddit_bars = []
    lin_twitter_bars = []
    res_reddit_bars = []
    res_twitter_bars = []
    path_reddit_bars = []
    path_twitter_bars = []
    lesk_reddit_bars = []
    lesk_twitter_bars = []
    hso_reddit_bars = []
    hso_twitter_bars = []
    for topic in topics:
        wup_reddit_bars.append(reddit_data[topic][0])
        wup_twitter_bars.append(twitter_data[topic][0])
        jcn_reddit_bars.append(reddit_data[topic][1])
        jcn_twitter_bars.append(twitter_data[topic][1])
        lhc_reddit_bars.append(reddit_data[topic][2])
        lhc_twitter_bars.append(twitter_data[topic][2])
        lin_reddit_bars.append(reddit_data[topic][3])
        lin_twitter_bars.append(twitter_data[topic][3])
        res_reddit_bars.append(reddit_data[topic][4])
        res_twitter_bars.append(twitter_data[topic][4])
        path_reddit_bars.append(reddit_data[topic][5])
        path_twitter_bars.append(twitter_data[topic][5])
        lesk_reddit_bars.append(reddit_data[topic][6])
        lesk_twitter_bars.append(twitter_data[topic][6])
        hso_reddit_bars.append(reddit_data[topic][7])
        hso_twitter_bars.append(twitter_data[topic][7])


    # Set position of bar on X axis
    ind = np.arange(len(topics))  # the x locations for the groups
    width = 0.2
    plt.subplot(2, 4, 1)
    plt.bar(ind, wup_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, wup_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.ylabel('Value', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('WUP')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 2)
    plt.bar(ind, jcn_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, jcn_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('JCN')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 3)
    plt.bar(ind, lhc_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, lhc_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('LHC')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 4)
    plt.bar(ind, lin_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, lin_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('LIN')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 5)
    plt.bar(ind, res_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, res_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.ylabel('Value', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('RES')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 6)
    plt.bar(ind, path_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, path_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('PATH')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 7)
    plt.bar(ind, lesk_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, lesk_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('LESK')
    plt.grid()
    plt.legend()

    plt.subplot(2, 4, 8)
    plt.bar(ind, hso_reddit_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, hso_twitter_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.xticks([r + width for r in range(len(topics))], topics)

    # Create legend & Show graphic
    plt.title('HSO')
    plt.grid()
    plt.legend()

    plt.show()

def plot_temporal_analysis(r_topics_rates, t_topics_rates):
    reddit_bars = []
    twitter_bars = []
    r_topics = []
    t_topics = []
    for tuple in r_topics_rates:
        reddit_bars.append(tuple[1])
        r_topics.append(tuple[0])
    for tuple in t_topics_rates:
        twitter_bars.append(tuple[1])
        t_topics.append(tuple[0])

    # Set position of bar on X axis
    ind = np.arange(len(r_topics))  # the x locations for the groups
    width = 0.2

    plt.subplot(1, 2, 1)
    plt.bar(ind, reddit_bars, width, color='orange', label='Reddit Topics Rate')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.ylabel('Topic Rate', fontweight='bold')
    plt.xticks([r + width for r in range(len(r_topics))], r_topics, rotation=30, ha='right')

    # Create legend & Show graphic
    plt.title('Percentage of Reddit comments of a certain topic')
    plt.grid()
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.bar(ind, twitter_bars, width, color='blue', label='Twitter Topics Rate')

    # Add xticks on the middle of the group bars
    plt.xlabel('Topics', fontweight='bold')
    plt.ylabel('Topic Rate', fontweight='bold')
    plt.xticks([r + width for r in range(len(t_topics))], t_topics, rotation=30, ha='right')

    # Create legend & Show graphic
    plt.title('Percentage of Twitter comments of a certain topic')
    plt.grid()
    plt.legend()

    plt.show()

def plot_emojis_cross_topic(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                            all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                            trimesters):
    reddit_all_emojis, reddit_all_kemojis = get_emojis_all_trimesters(all_reddit_topics, all_reddit_clusters,
                                                                      all_reddit_paths, trimesters)
    twitter_all_emojis, twitter_all_kemojis = get_emojis_all_trimesters(all_twitter_topics, all_twitter_clusters,
                                                                        all_twitter_paths, trimesters)
    r_emojis_bars = []
    r_k_emojis_bars = []
    t_emojis_bars = []
    t_k_emojis_bars = []
    for trimester in trimesters:
        r_emojis_bars.append(reddit_all_emojis[trimester])
        r_k_emojis_bars.append(reddit_all_kemojis[trimester])
        t_emojis_bars.append(twitter_all_emojis[trimester])
        t_k_emojis_bars.append(twitter_all_kemojis[trimester])

        # Set position of bar on X axis
    ind = np.arange(len(trimesters))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, r_k_emojis_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, t_k_emojis_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('Emojis mean', fontweight='bold')
    plt.xticks([r + width for r in range(len(trimesters))], trimesters)

    # Create legend & Show graphic
    plt.title('Mean number of keyboard emojis per comment')
    plt.grid()
    plt.legend()
    plt.show()

def plot_swearings_cross_topic(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                               all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                               trimesters):
    reddit_all_swearings = get_swearings_all_trimesters(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                                                        trimesters)
    twitter_all_swearings = get_swearings_all_trimesters(all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                                                         trimesters)

    r_swear_bars = []
    t_swear_bars = []
    for trimester in trimesters:
        r_swear_bars.append(reddit_all_swearings[trimester])
        t_swear_bars.append(twitter_all_swearings[trimester])

        # Set position of bar on X axis
    ind = np.arange(len(trimesters))  # the x locations for the groups
    width = 0.2
    plt.bar(ind, r_swear_bars, width, color='orange', label='Reddit')
    plt.bar(ind + width, t_swear_bars, width, color='blue', label='Twitter')

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('Swearing rate', fontweight='bold')
    plt.xticks([r + width for r in range(len(trimesters))], trimesters)

    # Create legend & Show graphic
    plt.title('Swearing words over the total number of words')
    plt.grid()
    plt.legend()
    plt.show()

def plot_indices_cross_topic(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                             all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                             trimesters):
    reddit_ari, reddit_cli = get_readability_all_trimesters(all_reddit_topics, all_reddit_clusters,
                                                            all_reddit_paths, trimesters)
    twitter_ari, twitter_cli = get_readability_all_trimesters(all_twitter_topics, all_twitter_clusters,
                                                              all_twitter_paths, trimesters)

    reddit_ari_bars = []
    reddit_cli_bars = []
    twitter_ari_bars = []
    twitter_cli_bars = []
    for trimester in trimesters:
        reddit_ari_bars.append(reddit_ari[trimester])
        reddit_cli_bars.append(reddit_cli[trimester])
        twitter_ari_bars.append(twitter_ari[trimester])
        twitter_cli_bars.append(twitter_cli[trimester])

    # Set position of bar on X axis
    ind = np.arange(len(trimesters))  # the x locations for the groups
    width = 0.18
    edge_width = 0.75
    bar1 = plt.bar(ind, reddit_ari_bars, width, color='orange', label='Reddit scores', edgecolor='black',
                   linewidth=edge_width)
    bar2 = plt.bar(ind + width, reddit_cli_bars, width, color='orange', edgecolor='black', linewidth=edge_width)
    bar3 = plt.bar(ind + 2 * width, twitter_ari_bars, width, color='blue', label='Twitter scores', edgecolor='black',
                   linewidth=edge_width)
    bar4 = plt.bar(ind + 3 * width, twitter_cli_bars, width, color='blue', edgecolor='black', linewidth=edge_width)

    height = bar1[0].get_height()
    plt.text(bar1[0].get_x() + bar1[0].get_width() / 2.0, height, 'ARI', ha='center', va='bottom', fontweight='bold')
    height = bar2[0].get_height()
    plt.text(bar2[0].get_x() + bar2[0].get_width() / 2.0, height, 'CLI', ha='center', va='bottom', fontweight='bold')
    height = bar3[0].get_height()
    plt.text(bar3[0].get_x() + bar3[0].get_width() / 2.0, height, 'ARI', ha='center', va='bottom', fontweight='bold')
    height = bar4[0].get_height()
    plt.text(bar4[0].get_x() + bar4[0].get_width() / 2.0, height, 'CLI', ha='center', va='bottom', fontweight='bold')

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('ARI and CLI scores', fontweight='bold')
    plt.xticks([r + width for r in range(len(trimesters))], trimesters)

    # Create legend & Show graphic
    plt.title('Scores of readability')
    plt.grid()
    plt.legend()
    plt.show()

def plot_relations_evolution(trimesters):
    r_2015_1st = [10.75, 5.21, 6.12, 13.21, 4.83, 7.14]
    r_2015_2nd = [4.37, 5.54, 5.46, 4.71, 3.94, 4.74]
    r_2016_1st = [7.59, 6.84, 8.26, 6.63, 5.95, 5.55]
    r_2016_2nd = [5.77, 6.40, 6.80, 5.40, 6.14, 4.85]
    r_2017_1st = [9.83, 7.28, 13.11, 5.76, 12.20, 5.09]
    r_2017_2nd = [25.55, 1.36, 13.06, 27.67, 1.31, 14.19]
    t_2015_1st = [0.15, 1.68, 0.09, 0.28, 0.42, 0.30]
    t_2015_2nd = [0.23, 0.30, 0.50, 0.30, 0.39, 0.37]
    t_2016_1st = [0.32, 0.34, 0.41, 0.33, 0.41]
    t_2016_2nd = [0.41, 0.41, 0.37, 0.33, 0.25, 0.42]
    t_2017_1st = [0.41, 0.66, 0.42, 0.45, 0.30, 0.35]
    t_2017_2nd = [0.38, 0.40, 0.51, 0.30, 0.65]

    reddit_bars = []
    reddit_bars.append(mean(r_2015_1st))
    reddit_bars.append(mean(r_2015_2nd))
    reddit_bars.append(mean(r_2016_1st))
    reddit_bars.append(mean(r_2016_2nd))
    reddit_bars.append(mean(r_2017_1st))
    reddit_bars.append(mean(r_2017_2nd))

    twitter_bars = []
    twitter_bars.append(mean(t_2015_1st))
    twitter_bars.append(mean(t_2015_2nd))
    twitter_bars.append(mean(t_2016_1st))
    twitter_bars.append(mean(t_2016_2nd))
    twitter_bars.append(mean(t_2017_1st))
    twitter_bars.append(mean(t_2017_2nd))

    plt.plot(trimesters, reddit_bars, color='orange', label='Reddit')
    plt.plot(trimesters, twitter_bars, color='blue', label='Twitter')
    #plt.ylim([0, 1])

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('Mean number of interactions', fontweight='bold')

    # Create legend & Show graphic
    plt.title('Mean number of interactions per trimester')
    plt.grid()
    plt.legend()
    plt.show()

def plot_specificity_evolution(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                               all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                               trimesters):
    r_specificity = get_trimesters_specificity(all_reddit_topics, all_reddit_clusters, all_reddit_paths, trimesters)
    t_specificity = get_trimesters_specificity(all_twitter_topics, all_twitter_clusters, all_twitter_paths, trimesters)
    reddit_bars = []
    twitter_bars = []
    for trimester in trimesters:
        reddit_bars.append(r_specificity[trimester])
        twitter_bars.append(t_specificity[trimester])

    plt.plot(trimesters, reddit_bars, color='orange', label='Reddit')
    plt.plot(trimesters, twitter_bars, color='blue', label='Twitter')
    #plt.ylim([0, 1])

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('Mean topics specificity', fontweight='bold')

    # Create legend & Show graphic
    plt.title('Mean topics specificity computed on each trimester')
    plt.grid()
    plt.legend()
    plt.show()

def plot_trimesters_adhesion(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                             all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                             trimesters):
    r_adhesion = get_trimesters_adhesions(all_reddit_topics, all_reddit_clusters, all_reddit_paths, trimesters)
    t_adhesion = get_trimesters_adhesions(all_twitter_topics, all_twitter_clusters, all_twitter_paths, trimesters)
    reddit_bars = []
    twitter_bars = []
    trimesters = trimesters[1:]
    for trimester in trimesters:
        reddit_bars.append(r_adhesion[trimester])
        twitter_bars.append(t_adhesion[trimester])
    plt.plot(trimesters, reddit_bars, color='orange', label='Reddit')
    plt.plot(trimesters, twitter_bars, color='blue', label='Twitter')
    # plt.ylim([0, 1])

    # Add xticks on the middle of the group bars
    plt.xlabel('Trimesters', fontweight='bold')
    plt.ylabel('Percentages', fontweight='bold')

    # Create legend & Show graphic
    plt.title('Percentage of topics changed with respect to the previous trimester')
    plt.grid()
    plt.legend()
    plt.show()

def plot_all_adhesions(all_reddit_topics, all_reddit_clusters, all_reddit_paths,
                       all_twitter_topics, all_twitter_clusters, all_twitter_paths,
                       trimesters):
    r_adhesion = get_all_topics_adhesions(all_reddit_topics, all_reddit_clusters, all_reddit_paths, trimesters)
    t_adhesion = get_all_topics_adhesions(all_twitter_topics, all_twitter_clusters, all_twitter_paths, trimesters)

    width = 0.2
    plt.bar(0, r_adhesion, width, color='orange', label='Reddit')
    plt.bar(0 + width, t_adhesion, width, color='blue', label='Twitter')
    plt.xlim([-0.6, 0.6])

    # Add xticks on the middle of the group bars
    plt.xlabel('Platform', fontweight='bold')
    plt.ylabel('Topics adhesion', fontweight='bold')
    plt.xticks([0, 0.2], ['Reddit', 'Twitter'])

    # Create legend & Show graphic
    plt.title("Topics' adhesion over the trimesters")
    plt.grid()
    plt.legend()
    plt.show()




if __name__ == '__main__':
    
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
    #topic_list = ['Music']
    topic_list = ['Smartphones', 'Sex', 'Body-care', 'Education', 'Books', 'Money', 'Racism', 'Sexual-orientation', 'Baseball', 'Fishing', 'Sleeping', 'Music']
    topic_labels = ['Smartphones', 'Human-Relationships', 'Body-care', 'Education', 'Books', 'Money', 'Racism', 'Sexual-orientation', 'Baseball', 'Fishing', 'Sleeping', 'Music']

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

    #plot_emojis(reddit_dict, twitter_dict, topic_list, topic_labels, K=False)
    #plot_emojis(reddit_dict, twitter_dict, topic_list, topic_labels, K=True)
    plot_read_indices(reddit_dict, twitter_dict, topic_list, topic_labels)
    #plot_uppercase_char_rate(reddit_dict, twitter_dict, topic_list)
    #plot_uppercase_word_rate(reddit_dict, twitter_dict, topic_list, topic_labels)
    #plot_oov_graph(reddit_dict, twitter_dict, 500, 5001, 500, topic_list)
    #plot_oov_bars(reddit_dict, twitter_dict, topic_list, topic_labels)
    #plot_swearings(reddit_dict, twitter_dict, topic_list, topic_labels)
    #twitter_emojis, twitter_k_emojis = get_all_emojis(twitter_dict, topic_list)
    #swearing_count(reddit_dict, topic_list)
    #uppercase_rate(twitter_dict, topic_list)
    #uppercase_words(twitter_dict, topic_list)
    #print(reddit_dict['Body-care'])
    #plot_reading_time(reddit_dict, twitter_dict, topic_list, topic_labels)


