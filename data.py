'''
Classes/functions to read and featurize data
'''
import argparse
import logging
import time
import os
import pandas as pd
from sklearn import preprocessing
import torch
from transformers import BertTokenizer, BertModel
import numpy as np
from tqdm import tqdm
import pickle
from utils import get_appendix
from glob import glob

class Data:
    def __init__(self, raw_data, split, text_only=False, include_tfidf=False, balanced=False):
        self.bert = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = True).eval()
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        self.scaler = preprocessing.StandardScaler() # Can change this to choose different scaler
        self.split = split

        self.X, self.y = self.featurize(raw_data, text_only=text_only, include_tfidf=include_tfidf, balanced=balanced)


    # Code from https://mccormickml.com/2019/05/14/BERT-word-embeddings-tutorial/
    def _get_bert_embed(self, text):
        marked_text = "[CLS] " + text + " [SEP]"

        # Tokenize our sentence with the BERT tokenizer.
        tokenized_text = self.tokenizer.tokenize(marked_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)
        segments_ids = [1] * len(tokenized_text)

        tokens_tensor = torch.tensor([indexed_tokens])
        segments_tensors = torch.tensor([segments_ids])
 
        with torch.no_grad():
            outputs = self.bert(tokens_tensor, segments_tensors)
            hidden_states = outputs[2]

        token_vecs = hidden_states[-2][0]

        # Calculate the average of all token vectors.
        sentence_embedding = torch.mean(token_vecs, dim=0)

        return sentence_embedding.tolist()

    # Change this function to change how data is featurized
    def featurize(self, df, text_only=False, include_tfidf=False, balanced=False):
        if balanced:
            X = df.drop(columns=['label']).values
            y = df['label'].values
            return X, y
        if 'processed_text' not in df.columns:
            df['processed_text'] = df['text']
        df = df[~df['processed_text'].isna()].reset_index()
        if text_only:
            X = df['processed_text'].values
            y = df['label'].values
        else:    
            x1_cols = ['like_count', 'quote_count', 
                        'mentions_count', 'author_followers',
                        'sentiment_score_pos', 'sentiment_score_neu', 
                        'sentiment_score_neg', 'sentiment_score_comp',
                        'text_tfid_sum', 'text_tfid_max', 'text_tfid_min', 'text_tfid_avg',
                        'text_tfid_std', 'hashtag_tfid_sum', 'hashtag_tfid_max',
                        'hashtag_tfid_min', 'hashtag_tfid_avg', 'hashtag_tfid_std']
            X1 = df[x1_cols]
            X1 = self.scaler.fit_transform(X1) # Scale values
            X1 = pd.DataFrame(X1, columns=x1_cols)

            logging.info('Getting bert embeddings...')
            X2 = pd.DataFrame([self._get_bert_embed(tweet) for tweet in tqdm(df['processed_text'].values)])
            X2.columns = [f'b{i}' for i in range(X2.shape[1])]


            if include_tfidf:
                logging.info('Loading precomputed tfidf scores')
                X3 = pd.DataFrame([eval(row) for row in tqdm(df['hashtags_tfidf'].values)])
                X3.columns = [f'h{i}' for i in range(X3.shape[1])]
                X = pd.concat([X1, X2, X3], axis=1, sort=False)
            else:
                X = pd.concat([X1, X2], axis=1, sort=False)

            X = X.values
            y = df['label'].values

        return X, y
    
    def to_pickle(self, outpath):
        with open(outpath, "wb") as f:
            pickle.dump(self, f)

def read_pickle(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data


# Loads and caches data, does not cache if text_only features, 
# because this is model dependent and quick to compute
def load(datadir, cachedir=None, override_cache=False, 
        text_only=False, include_tfidf=False, balanced=False):
    # If exists caches for all splits and not override cache, load cached data
    appendix = get_appendix(include_tfidf, balanced)
    if all((
            os.path.exists(os.path.join(cachedir, f'train{appendix}.pkl')),
            os.path.exists(os.path.join(cachedir, f'dev{appendix}.pkl')),
            os.path.exists(os.path.join(cachedir, f'test{appendix}.pkl')),
            not override_cache,
            not text_only
        )):
        train = read_pickle(os.path.join(cachedir, f'train{appendix}.pkl'))
        dev  = read_pickle(os.path.join(cachedir, f'dev{appendix}.pkl'))
        test = read_pickle(os.path.join(cachedir, f'test{appendix}.pkl'))


    else:
        if balanced:
            train = pd.read_csv(glob(os.path.join(datadir, 'balanced', '*train*.csv'))[0])
            dev = pd.read_csv(glob(os.path.join(datadir, 'balanced', '*dev*.csv'))[0])
            test = pd.read_csv(glob(os.path.join(datadir, 'balanced', '*test*.csv'))[0])

            train = Data(train, "train", text_only=text_only, include_tfidf=include_tfidf, balanced=balanced)
            dev = Data(dev, "dev", text_only=text_only, include_tfidf=include_tfidf, balanced=balanced)
            test = Data(test, "test", text_only=text_only, include_tfidf=include_tfidf, balanced=balanced)

        else:
            all_data = pd.read_csv(os.path.join(datadir, 'all_data_preprocessed.tsv'), sep='\t')
            all_data.drop(columns=['retweet_count', 'reply_count'], inplace=True)

            # Read splits
            train_ids = pd.read_csv(os.path.join(datadir, 'train-ids.txt'), names=['id'])
            dev_ids = pd.read_csv(os.path.join(datadir, 'dev-ids.txt'), names=['id'])
            test_ids = pd.read_csv(os.path.join(datadir, 'test-ids.txt'), names=['id'])

            train = Data(train_ids.merge(all_data, how='inner', on='id'), "train", text_only=text_only, include_tfidf=include_tfidf)
            dev = Data(dev_ids.merge(all_data, how='inner', on='id'), "dev", text_only=text_only, include_tfidf=include_tfidf)
            test = Data(test_ids.merge(all_data, how='inner', on='id'), "test", text_only=text_only, include_tfidf=include_tfidf)

        if cachedir and not text_only:
            train.to_pickle(os.path.join(cachedir, f'train{appendix}.pkl'))
            dev.to_pickle(os.path.join(cachedir, f'dev{appendix}.pkl'))
            test.to_pickle(os.path.join(cachedir, f'test{appendix}.pkl'))

    return train, dev, test

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', default='data')
    parser.add_argument('--cachedir', default='data')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    # Dataframes
    train, dev, test = load(args.datadir, cachedir=args.cachedir, override_cache=True)
    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')