'''
Reads in all datafile and output train/dev/test splits
'''

import argparse
import logging
import time
import random
import pandas as pd
import os
import numpy as np
from imblearn.over_sampling import SMOTE 
from collections import Counter

random.seed(1)

def save_ids(ids, outpath):
    ids = [str(i) for i in ids]
    with open(outpath, 'w') as f:
        f.write('\n'.join(ids))

def read_and_split_data(datasplit, datafile, outdir, balanced=False):
    # Process split
    train_split, dev_split, test_split = [float(d)/100 for d in datasplit.split('/')]
    assert (train_split + dev_split + test_split) == 1.

    # Open data file
    df = pd.read_csv(datafile, sep='\t', dtype=str)
    all_ids = list(df['id'])
    all_labels = list(df['label'])
    logging.info(f'Read {len(all_ids)} ids.')
    random.shuffle(all_ids)

    # Split data
    s1 = int(len(all_ids)*train_split)
    s2 = s1 + int(len(all_ids)*dev_split)
    train, dev, test = all_ids[:s1], all_ids[s1:s2], all_ids[s2:]
    train_labels, dev_labels, test_labels = all_labels[:s1], all_labels[s1:s2], all_labels[s2:]


    logging.info(f'Num ids in train: {len(train)}')
    logging.info(f'Num ids in dev: {len(dev)}')
    logging.info(f'Num ids in test: {len(test)}')
    save_ids(train, os.path.join(outdir, 'train-ids.txt'))
    save_ids(dev, os.path.join(outdir, 'dev-ids.txt'))
    save_ids(test, os.path.join(outdir, 'test-ids.txt'))

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasplit', default="60/20/20", help='Split percentages, e.g. 60/20/20')
    parser.add_argument('--datafile', default="../data/all_data_preprocessed.tsv", help='Path to datafile')
    parser.add_argument('--outdir', default="../data", help='Path to outdir')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')

    start = time.time()
    
    read_and_split_data(args.datasplit, args.datafile, args.outdir)

    end = time.time()
    logging.info(f'Time to run script: {end-start} secs')