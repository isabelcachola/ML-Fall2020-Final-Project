""" Main function, which trains our model and makes predictions with it. """
import csv
import argparse as ap
import logging

import numpy as np
import torch
from sklearn.metrics import classification_report
from data import load
from models import FeedForward, BiLSTM, LogisticRegression
from utils import train_pytorch, test_pytorch
from pprint import pprint
import json

def get_args():
    """ Define our command line arguments. """
    p = ap.ArgumentParser()

    # Mode to run the model in.
    p.add_argument("mode", choices=["train", "predict"], type=str)
    p.add_argument("model", type=str, default="simple-ff")

    # File locations 
    p.add_argument("--data-dir", type=str, default="data")
    p.add_argument("--log-file", type=str, default="logs.csv")
    p.add_argument('--cachedir', type=str, default="data")
    p.add_argument('--balanced', default=False, action='store_true', help='Flag to use balanced data')
    p.add_argument('--override-cache', default=False, action='store_true')


    # hyperparameters
    p.add_argument("--batch-size", type=int, default=36)

    # simple-ff hparams
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--ff-hunits", type=int, default=128)

    # bi-lstm hparams
    p.add_argument("--num-epochs", type=int, default=5)
    

    return p.parse_args()


def train(args):
    # load data
    train, dev, _ = load(args.data_dir, cachedir=args.cachedir, 
                        override_cache=args.override_cache, 
                        text_only=(args.model.lower() == "bi-lstm"))
    train_data, train_labels = train.X, train.y
    dev_data, dev_labels = dev.X, dev.y

    # Build model
    if args.model.lower() == "simple-ff":
        model = FeedForward(args.ff_hunits, train.X.shape[1])
        train_pytorch(args, model, train_data, train_labels, dev_data, dev_labels)
    elif args.model.lower() == "bi-lstm":
        model = BiLSTM(epochs=args.num_epochs, batch_size=args.batch_size)
        model.train(train_data, train_labels, dev_data, dev_labels)
    elif args.model.lower() == "logreg":
        model = LogisticRegression()
        model.train(train_data, train_labels, dev_data, dev_labels)
    else:
        raise Exception("Unknown model type passed in!")
    

def test(args):
    _, _, test = load(args.data_dir, cachedir=args.cachedir, 
                    override_cache=args.override_cache, 
                    text_only=(args.model.lower() == "bi-lstm"))
    test_data, test_labels = test.X, test.y

    if args.model.lower() == "simple-ff":
        preds = test_pytorch(test_data, test_labels)
    elif args.model.lower() == "bi-lstm":
        model = BiLSTM(load_model_path="models/bilstm.keras", tokenizer_path='models/bilstm-tokenizer.json')
        preds = model.test(test_data, y_test=test_labels)
    elif args.model.lower() == "logreg":
        model = LogisticRegression(load_model_path="models/logreg.pkl")
        preds = model.test(test_data, test_labels)
    else:
        raise Exception("Unknown model type passed in!")
    
    metrics = classification_report(test_labels, preds, output_dict=True)
    pprint(metrics)
    with open(f"scores/{args.model.lower()}.json", "w") as fout:
        json.dump(metrics, fout, indent=4)



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
    ARGS = get_args()
    if ARGS.mode == 'train':
        train(ARGS)
    elif ARGS.mode == 'predict':
        test(ARGS)
    else:
        print(f'Invalid mode: {ARGS.mode}! Must be either "train" or "predict".')
