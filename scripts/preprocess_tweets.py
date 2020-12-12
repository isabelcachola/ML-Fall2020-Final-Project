"""
Script to process tweets into featurized format
"""
import argparse
import logging
import time
import os
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import pandas as pd
import texthero as hero
import glob
import tqdm
import csv
import requests
from CREDS import BEARER_TOKEN


def getUserNumOfFollowers(df):
    """
    This function returns a dataframe with the number of followes of the author of each
    tweet in the given dataframe
    :param df: The dataframe with the information about the desired tweets
    :return: the df dataframe with an extra column containing the number of followers of the
    tweets' author
    """

    headers = {"Authorization": "Bearer {}".format(BEARER_TOKEN)}
    userIds = df["author_id"].tolist()
    index = 0
    step = 100  # Number of ids to request at once
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
        time.sleep(2)  # TODO: Why do we have to sleep here?
    pbar.close()
    return df


def getSentimentScore(text, sentiment_type='compound'):
    """
    This function extracts the sentiment score of the given text
    :param text: the text we want to extract the sentiment score from
    :param sentiment_type: the type of sentiment score we want (positive, negative, neutral, or compound)
    :return: the sentiment score
    """

    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    return ss[sentiment_type]


def getTfidfStats(X_raw):
    """
    This function gets the stats of the tfidf of the hashtags and text of the tweets
    in the given dataframe
    :param X_raw: the given dataframe with the tweets information
    :return: X_raw with columns containing information about the tfidf of the hashtags
    and text of the tweets
    """

    vec = TfidfVectorizer()
    vec_result = vec.fit_transform(X_raw["processed_text"])
    root_text_data = pd.DataFrame(vec_result.toarray(), columns=vec.get_feature_names())

    X_raw["text_tfid_sum"] = root_text_data.sum(axis=1)
    X_raw["text_tfid_max"] = root_text_data.max(axis=1)
    X_raw["text_tfid_min"] = root_text_data.min(axis=1)
    X_raw["text_tfid_avg"] = root_text_data.mean(axis=1)
    X_raw["text_tfid_std"] = root_text_data.std(axis=1)

    # Getting the tfidf from the hashtags
    vec_hash = TfidfVectorizer()
    X_raw["hashtags"] = X_raw["hashtags"].fillna("")
    vec_result_hash = vec_hash.fit_transform(X_raw["hashtags"])
    root_hash_data = pd.DataFrame(vec_result_hash.toarray(), columns=vec_hash.get_feature_names())

    X_raw["hashtag_tfid_sum"] = root_hash_data.sum(axis=1)
    X_raw["hashtag_tfid_max"] = root_hash_data.max(axis=1)
    X_raw["hashtag_tfid_min"] = root_hash_data.min(axis=1)
    X_raw["hashtag_tfid_avg"] = root_hash_data.mean(axis=1)
    X_raw["hashtag_tfid_std"] = root_hash_data.std(axis=1)
    return X_raw


def filterSentences(sentence):
    """
    This function tokenizes the sentences and removes non-words and special characters
    or numbers
    :param sentence:the sentence to be tokenized
    :return the processed sentence
    """

    # Erasing nonwords from text
    words = set(nltk.corpus.words.words())
    sent = " ".join(w for w in nltk.wordpunct_tokenize(sentence) if w.lower() in words)
    return sent


def preprocess_tweets(datadir):
    """
    This function preprocessed the tweets information so it is in the format used by our ml
    models
    :param datadir: the directory where the root tweet information is
    :return: the formatted tweet information in dataframe format
    """

    files = glob.glob(os.path.join(os.path.abspath(datadir), "*.tsv"))
    logging.info(f'Found {len(files)} files')

    dfs = []
    for f in tqdm.tqdm(files, desc="Reading data"):
        dfs.append(pd.read_csv(f, sep='\t'))
    raw_data = pd.concat(dfs).drop_duplicates(subset=['id'])

    id_list = []
    text_list = []
    author_id_list = []
    retweet_count_list = []
    reply_count_list = []
    like_count_list = []
    quote_count_list = []
    mentions_list = []
    mentions_count_list = []
    hashtags_list = []
    label_list = []

    for row_idx, row in tqdm.tqdm(raw_data.iterrows(), 'Formatting data', total=len(raw_data)):
        # Getting new info
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

        # Appending to list
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
    df["sentiment_score_pos"] = df['text'].apply(getSentimentScore, sentiment_type="pos")
    df["sentiment_score_neu"] = df['text'].apply(getSentimentScore, sentiment_type="neu")
    df["sentiment_score_neg"] = df['text'].apply(getSentimentScore, sentiment_type="neg")
    df["sentiment_score_comp"] = df['text'].apply(getSentimentScore, sentiment_type="compound")

    logging.info('Getting tfidf stats')
    df = getTfidfStats(df)

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', '-d', dest='datadir', type=str, help='Directory to store data',
                        default="../data/raw_data/")
    parser.add_argument('--outdir', '-o', dest='outdir', type=str, help='Directory to store data', default="../data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()

    df = preprocess_tweets(args.datadir)
    outpath = os.path.join(args.outdir, 'all_data_preprocessed.tsv')
    df.to_csv(outpath, sep='\t', quoting=csv.QUOTE_NONNUMERIC, index=False)

    end = time.time()
    logging.info(f'Time to run script: {end - start} secs')
