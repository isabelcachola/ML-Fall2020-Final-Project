'''
Script to process tweets into featurized format
'''
import argparse
import logging
import time
import os
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd
import texthero as hero
import glob
import tqdm
import csv
import json

def getSentimentScore(text, type='compound'):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    return ss[type]

def preprocess_tweets(datadir):
    files = glob.glob(os.path.join(os.path.abspath(datadir), "*.tsv"))
    logging.info(f'Found {len(files)} files')

    dfs = []

    for f in tqdm.tqdm(files, desc="Reading data"):
        dfs.append(pd.read_csv(f, sep='\t'))
    
    raw_data = pd.concat(dfs).drop_duplicates(subset=['id'])
    # before = len(raw_data)
    # raw_data.dropna(inplace=True, subset=['entities'])
    # after = len(raw_data)
    # logging.info(f'Dropped {before-after} rows with NA values. len(raw_data)={len(raw_data)}')

    id_list = []
    text_list = []
    author_id_list = []
    # author_num_followers_list = []
    retweet_count_list = []
    reply_count_list = []
    like_count_list = []
    quote_count_list = []
    mentions_list = []
    hashtags_list = []
    label_list = []

    for row_idx, row in tqdm.tqdm(raw_data.iterrows(), 'Formatting data', total=len(raw_data)):
        #Getting new info
        cur_id = row['id']

        # Remove special characters
        formattex_text = re.sub(r"[^a-zA-Z0-9]+", ' ', row['text'])

        row['public_metrics'] = eval(row['public_metrics'])
        # import ipdb;ipdb.set_trace()
        try:
            row['entities'] = eval(row['entities'])
        except:
            row['entities'] = []

        # import ipdb; ipdb.set_trace()
        cur_author_id = row['author_id']
        # curr_author_num_followers = row['public_metrics']['user-followers_count']
        cur_retweet_count = row['public_metrics']['retweet_count']
        cur_reply_count = row['public_metrics']['reply_count']
        cur_like_count = row['public_metrics']['like_count']
        cur_quote_count = row['public_metrics']['quote_count']
        curr_label = 1 if (cur_reply_count > cur_retweet_count) else 0
        cur_mentions = []
        try:
            for mention in row['entities']['mentions']:
                cur_mentions.append(mention['username'])
        except:
            cur_mentions = []

        cur_hashtags = ""
        # import ipdb;ipdb.set_trace()
        try:
            for hashtag in row['entities']['hashtags']:
                cur_hashtags = cur_hashtags + " " + hashtag['tag']
        except:
            cur_hashtags = ""

        #Appending to list
        id_list.append(cur_id)
        text_list.append(formattex_text)
        author_id_list.append(cur_author_id)
        retweet_count_list.append(cur_retweet_count)
        reply_count_list.append(cur_reply_count)
        like_count_list.append(cur_like_count)
        quote_count_list.append(cur_quote_count)
        mentions_list.append(cur_mentions)
        hashtags_list.append(cur_hashtags)
        label_list.append(curr_label)

    data = {'id': id_list, 'text': text_list, 'author_id': author_id_list, 
            # 'author_followers':author_num_followers_list,
            'retweet_count': retweet_count_list, 'reply_count': reply_count_list,
            'like_count': like_count_list, 'quote_count': quote_count_list,
            'mentions': mentions_list, 'hashtags': hashtags_list, 'label': label_list}

    logging.info('Creating dataframe')
    df = pd.DataFrame(data=data)
    logging.info('Gettting hashtag tfidf')
    df["hashtags_tfidf"] = hero.tfidf(df['hashtags'])
    logging.info('Getting sentiment scores')
    df["sentiment_score_pos"] = df['text'].apply(getSentimentScore, type="pos")
    df["sentiment_score_neu"] = df['text'].apply(getSentimentScore, type="neu")
    df["sentiment_score_neg"] = df['text'].apply(getSentimentScore, type="neg")
    df["sentiment_score_comp"] = df['text'].apply(getSentimentScore, type="compound")
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', dest='datadir', type=str, help='Directory to store data', default="../data/raw_data/")
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, help='Directory to store data', default="../data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filemode='a', format='%(levelname)s - %(message)s')

    start = time.time()

    df = preprocess_tweets(args.datadir)
    outpath = os.path.join(args.outdir, 'all_data_preprocessed.tsv')
    df.to_csv(outpath, sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)
    
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')