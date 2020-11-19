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

def getSentimentScore(text, type='compound'):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    return ss[type]

def preprocess_tweets(datadir):
    files = glob.glob(os.path.join(os.path.abspath(datadir), "*.tsv"))
    logging.info(f'Found {len(files)} files')

    # Read in total list of tweet ids and convert to int
    # This is useful to make sure you have all of the data
    total_id_list = list(map(lambda x: int(x.strip()), open(os.path.join(os.path.abspath(datadir), "all_tweet_ids.txt")).readlines()))

    dfs = []

    for f in tqdm.tqdm(files):
        dfs.append(pd.read_csv(f, sep='\t'))
    
    raw_data = pd.concat(dfs).drop_duplicates(subset=['id'])

    if len(raw_data) != len(total_id_list):
        logging.warning(f'Length of id list ({len(id_list)}) does not match number of tweets ({len(raw_data)}). \
            Make sure you have the most recent list of ids and data files.')

    id_list = []
    text_list = []
    author_id_list = []
    author_num_followers_list = []
    retweet_count_list = []
    reply_count_list = []
    like_count_list = []
    quote_count_list = []
    mentions_list = []
    hashtags_list = []

    for row_idx, row in raw_data.iterrows():
        #Getting new info
        cur_id = row['id']

        # Remove special characters
        formattex_text = re.sub(r"[^a-zA-Z0-9]+", ' ', row['text'])

        cur_author_id = row['user-id']
        curr_author_num_followers = row['user-followers_count']
        cur_retweet_count = row['retweet_count']
        cur_reply_count = row['reply_count']
        cur_like_count = row['favorite_count']
        cur_quote_count = row['quote_count']
        cur_mentions = []
        try:
            for mention in eval(row['entities-user_mentions']):
                cur_mentions = cur_mentions + [mention['username']]
        except:
            cur_mentions = []

        cur_hashtags = ""
        try:
            for hashtag in eval(row['entities-hashtags']):
                cur_hashtags = cur_hashtags + " " + hashtag['text']
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
        author_num_followers_list.append(curr_author_num_followers)

    data = {'id': id_list, 'text': text_list, 'author_id': author_id_list, 'author_followers':author_num_followers_list,
            'retweet_count': retweet_count_list, 'reply_count': reply_count_list,
            'like_count': like_count_list, 'quote_count': quote_count_list,
            'mentions': mentions_list, 'hashtags': hashtags_list}

    df = pd.DataFrame(data=data)
    df["sentiment_score_pos"] = df['text'].apply(getSentimentScore, type="pos")
    df["sentiment_score_neu"] = df['text'].apply(getSentimentScore, type="neu")
    df["sentiment_score_neg"] = df['text'].apply(getSentimentScore, type="neg")
    df["sentiment_score_comp"] = df['text'].apply(getSentimentScore, type="compound")
    df["hashtags_tfidf"] = hero.tfidf(df['hashtags'])
    return df


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', dest='datadir', type=str, help='Directory to store data', default="../data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, filemode='a', format='%(levelname)s - %(message)s')

    start = time.time()

    df = preprocess_tweets(args.datadir)
    outpath = os.path.join(args.datadir, 'all_data_preprocessed.tsv')
    df.to_csv(outpath, sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)
    
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')