'''
Script to process tweets into featurized format
'''
import argparse
import logging
import time
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re
import pandas as pd
import texthero as hero
import glob
import tqdm
import csv
import json
import requests
from CREDS import BEARER_TOKEN

# Gi's function from notebook, Isabel added a progress bar
def getUserNumOfFollowers(df):
    headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}
    userIds = df["author_id"].tolist()
    index = 0
    step = 100 # Number of ids to request at once
    pbar = tqdm.tqdm(total=len(userIds))
    while len(userIds) > index:
        end_index = index + step
        ids = []
        if end_index >= len(userIds):
            ids = userIds[index:]
        else:
            ids = userIds[index:end_index]
        inputs = {'ids': ','.join(str(e) for e in ids), 'user.fields': 'public_metrics'}
        response = requests.get("https://api.twitter.com/2/users", headers=headers,
                                params=inputs)
        if response.status_code != 200:
            raise Exception(
                "Cannot get stream (HTTP {}): {}".format(
                    response.status_code, response.text
                )
            )
        data_root = response.json()
        data_list = data_root['data']
        for item in data_list:
            cur_id = item['id']
            df.loc[df['author_id'] == int(cur_id), 'author_followers'] = item['public_metrics']['followers_count']
        index = end_index
        pbar.update(step)
        time.sleep(2)
    pbar.close()
    return df

# Gi's function
def getSentimentScore(text, type='compound'):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    return ss[type]

# Gi's function from notebook
def filterSentences(sentence):
  '''
  This function tokenizes the sentences for the use with bert
  Input:
    sentence: the sentence to be tokenized
  '''

  ## Erasing nonwords from text
  words = set(nltk.corpus.words.words())
  sent = " ".join(w for w in nltk.wordpunct_tokenize(sentence) if w.lower() in words)

  return sent

def preprocess_tweets(datadir):
    files = glob.glob(os.path.join(os.path.abspath(datadir), "*.tsv"))
    logging.info(f'Found {len(files)} files')

    dfs = []

    for f in tqdm.tqdm(files, desc="Reading data"):
        dfs.append(pd.read_csv(f, sep='\t'))
    
    raw_data = pd.concat(dfs).drop_duplicates(subset=['id'])

    id_list = []
    text_list = []
    author_id_list = []
    # author_num_followers_list = []
    retweet_count_list = []
    reply_count_list = []
    like_count_list = []
    quote_count_list = []
    mentions_list = []
    mentions_count_list = []
    hashtags_list = []
    label_list = []

    for row_idx, row in tqdm.tqdm(raw_data.iterrows(), 'Formatting data', total=len(raw_data)):
        #Getting new info
        cur_id = row['id']

        # Remove special characters
        formattex_text = re.sub(r"[^a-zA-Z0-9]+", ' ', row['text'])

        row['public_metrics'] = eval(row['public_metrics'])
        try:
            row['entities'] = eval(row['entities'])
        except:
            row['entities'] = []

        cur_author_id = row['author_id']
        cur_retweet_count = row['public_metrics']['retweet_count']
        cur_reply_count = row['public_metrics']['reply_count']
        cur_like_count = row['public_metrics']['like_count']
        cur_quote_count = row['public_metrics']['quote_count']
        curr_label = 1 if (cur_reply_count > cur_retweet_count) else 0
        # curr_num_followers = getUserNumOfFollowers(X_raw)
        cur_mentions = []
        try:
            for mention in row['entities']['mentions']:
                cur_mentions.append(mention['username'])
        except:
            cur_mentions = []

        cur_hashtags = ""
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
        mentions_count_list.append(len(cur_mentions))
        hashtags_list.append(cur_hashtags)
        label_list.append(curr_label)

    data = {'id': id_list, 'text': text_list, 'author_id': author_id_list, 
            # 'author_followers':author_num_followers_list,
            'retweet_count': retweet_count_list, 'reply_count': reply_count_list,
            'like_count': like_count_list, 'quote_count': quote_count_list,
            'mentions': mentions_list, 'mentions_count': mentions_count_list,
            'hashtags': hashtags_list, 'label': label_list}

    logging.info('Creating dataframe')
    df = pd.DataFrame(data=data)

    logging.info('Getting author num followers')
    df["author_followers"] = 0
    df = getUserNumOfFollowers(df)

    logging.info('Cleaning text')
    # Process the text to eliminate invalid characters and nonwords
    df["processed_text"] = df["text"].apply(filterSentences)

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

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()

    df = preprocess_tweets(args.datadir)
    outpath = os.path.join(args.outdir, 'all_data_preprocessed.tsv')
    df.to_csv(outpath, sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)
    
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')