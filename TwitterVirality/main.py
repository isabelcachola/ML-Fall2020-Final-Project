import requests
import os
import json
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
import re
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import texthero as hero

# Uncomment this like the first time you are running the code
# nltk.download('vader_lexicon')
headers = {"Authorization": "Bearer {}".format("[BEARER TOKEN]")}


def getTweetIdsWithQuery(query, numRequests=1):
    # Check if numRequests makes sense
    if numRequests < 0:
        raise Exception("Number of requests has to be greater than 0")

    if numRequests > 1000:
        raise Exception("Number of requests currently cannot be grater than 1000")

    # Get first results
    tweet_ids = []
    parameters = {"query": query}
    response = requests.get("https://api.twitter.com/2/tweets/search/recent",
                            headers=headers, params=parameters)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    data_root = response.json()
    data_list = data_root['data']
    next_token = data_root['meta']['next_token']
    for tweet in data_list:
        tweet_ids = tweet_ids + [tweet['id']]

    # Get next pages
    for request_index in range(numRequests - 1):
        parameters["next_token"] = next_token
        response = requests.get("https://api.twitter.com/2/tweets/search/recent",
                                headers=headers, params=parameters)
        if response.status_code != 200:
            raise Exception(
                "Cannot get stream (HTTP {}): {}".format(
                    response.status_code, response.text
                )
            )
        data_root = response.json()
        data_list = data_root['data']
        next_token = data_root['meta']['next_token']
        for tweet in data_list:
            tweet_ids = tweet_ids + [tweet['id']]

    return tweet_ids


def getTweetDetailsWithId(id):
    inputs = {'ids': ",".join(id), 'tweet.fields': 'author_id,public_metrics,entities',
              'expansions': 'entities.mentions.username'}
    response = requests.get("https://api.twitter.com/2/tweets", headers=headers,
                            params=inputs)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )

    data_root = response.json()
    data_list = data_root['data']
    return data_list


def getUserNumOfFollowers(userId):
    inputs = {'ids': userId, 'user.fields': 'public_metrics'}
    response = requests.get("https://api.twitter.com/2/users", headers=headers,
                            params=inputs)
    if response.status_code != 200:
        raise Exception(
            "Cannot get stream (HTTP {}): {}".format(
                response.status_code, response.text
            )
        )
    data_root = response.json()
    data_list = data_root['data'][0]
    return data_list['public_metrics']['followers_count']


def getSentimentScore(text, type='compound'):
    sid = SentimentIntensityAnalyzer()
    ss = sid.polarity_scores(text)
    return ss[type]

def getTfidf(word_list):
    if len(word_list) == 0:
        return [0]
    v = TfidfVectorizer()
    result = v.fit_transform(word_list)
    return result.toarray()


def getTweetData(query, numRequests=1):
    id_list = []
    text_list = []
    author_id_list = []
    retweet_count_list = []
    reply_count_list = []
    like_count_list = []
    quote_count_list = []
    mentions_list = []
    hashtags_list = []

    ids = getTweetIdsWithQuery(query, numRequests)
    root_dict_list = getTweetDetailsWithId(ids)

    for root_twitter_dict in root_dict_list:
        #Getting new info
        cur_id = root_twitter_dict['id']
        formattex_text = re.sub(r"[^a-zA-Z0-9]+", ' ', root_twitter_dict['text'])
        cur_author_id = root_twitter_dict['author_id']
        cur_retweet_count = root_twitter_dict['public_metrics']['retweet_count']
        cur_reply_count = root_twitter_dict['public_metrics']['reply_count']
        cur_like_count = root_twitter_dict['public_metrics']['like_count']
        cur_quote_count = root_twitter_dict['public_metrics']['quote_count']
        cur_mentions = []
        try:
            for mention in root_twitter_dict['entities']['mentions']:
                cur_mentions = cur_mentions + [mention['username']]
        except:
            cur_mentions = []

        cur_hashtags = ""
        try:
            for hashtag in root_twitter_dict['entities']['hashtags']:
                cur_hashtags = cur_hashtags + " " + hashtag['tag']
        except:
            cur_hashtags = ""

        #Appending to list
        id_list = id_list + [cur_id]
        text_list = text_list + [formattex_text]
        author_id_list = author_id_list + [cur_author_id]
        retweet_count_list = retweet_count_list + [cur_retweet_count]
        reply_count_list = reply_count_list + [cur_reply_count]
        like_count_list = like_count_list + [cur_like_count]
        quote_count_list = quote_count_list + [cur_quote_count]
        mentions_list = mentions_list + [cur_mentions]
        hashtags_list = hashtags_list + [cur_hashtags]

    data = {'id': id_list, 'text': text_list, 'author_id': author_id_list,
            'retweet_count': retweet_count_list, 'reply_count': reply_count_list,
            'like_count': like_count_list, 'quote_count': quote_count_list,
            'mentions': mentions_list, 'hashtags': hashtags_list}

    df = pd.DataFrame(data=data)
    df["author_followers"] = df['author_id'].apply(getUserNumOfFollowers)
    df["sentiment_score_pos"] = df['text'].apply(getSentimentScore, type="pos")
    df["sentiment_score_neu"] = df['text'].apply(getSentimentScore, type="neu")
    df["sentiment_score_neg"] = df['text'].apply(getSentimentScore, type="neg")
    df["sentiment_score_comp"] = df['text'].apply(getSentimentScore, type="compound")
    df["hashtags_tfidf"] = hero.tfidf(df['hashtags'])
    print(df)
    return df



#print(getSentimentScore("Im happy"))
#getUserNumOfFollowers(['528872520', '528872521'])

getTweetData("python")
