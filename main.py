""" Main function, which trains our model and makes predictions with it. """
import csv
import argparse as ap
import logging

import numpy as np
import torch
from sklearn.metrics import classification_report
from data import load
from models import FeedForward, BiLSTM, LogisticRegression, MajorityVote, SVM, Bert
from utils import train_pytorch, test_pytorch, get_appendix
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

    # Data params
    p.add_argument('--balanced', default=False, action='store_true', help='Flag to use balanced data')
    p.add_argument('--include-tfidf', default=False, action='store_true')
    p.add_argument('--override-cache', default=False, action='store_true')

    # hyperparameters
    p.add_argument("--batch-size", type=int, default=36)

    # simple-ff hparams
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--ff-hunits", type=int, default=128)

    # bi-lstm/bert hparams
    p.add_argument("--num-epochs", type=int, default=5)
    p.add_argument("--max-seq-len", type=int, default=25)
    

    return p.parse_args()


def train(args):
    # load data
    train, dev, _ = load(args.data_dir, cachedir=args.cachedir, 
                        override_cache=args.override_cache, 
                        text_only=(args.model.lower() in ["bi-lstm", "bert"]),
                        include_tfidf=args.include_tfidf,
                        balanced=args.balanced)
    train_data, train_labels = train.X, train.y
    dev_data, dev_labels = dev.X, dev.y

    # Build model
    apx = get_appendix(args.include_tfidf, args.balanced)
    if args.model.lower() == "simple-ff":
        model = FeedForward(args.ff_hunits, train.X.shape[1])
        train_pytorch(args, model,
                      train_data, train_labels,
                      dev_data, dev_labels,
                      save_model_path=f"models/simple-ff{apx}.torch")
    elif args.model.lower() == "bi-lstm":
        model = BiLSTM(epochs=args.num_epochs, 
                        batch_size=args.batch_size,
                        max_seq_len=args.max_seq_len)
        model.train(train_data, train_labels, dev_data, dev_labels)
    elif args.model.lower() == "logreg":
        model = LogisticRegression()
        model.train(train_data, train_labels, dev_data, dev_labels,
                    save_model_path=f"models/logreg{apx}.pkl")
    elif args.model.lower() == "majority-vote":
        model = MajorityVote()
        model.train(train_labels, dev_labels)
    elif args.model.lower() == "bert":
        model = Bert(epochs=args.num_epochs, 
                    batch_size=args.batch_size, 
                    max_seq_len=args.max_seq_len,
                    learning_rate=args.learning_rate
                    )
        model.train(train_data, train_labels, dev_data, dev_labels, 
                    save_model_path=f"models/bert.pkl")
    elif args.model.lower() == "svm":
        model = SVM()
        model.train(train_data, train_labels, save_model_path=f"models/svm{apx}.sav")
    else:
        raise Exception("Unknown model type passed in!")


def test(args):
    _, _, test = load(args.data_dir, cachedir=args.cachedir, 
                    override_cache=args.override_cache, 
                    text_only=(args.model.lower() in ["bi-lstm", "bert"]),
                    include_tfidf=args.include_tfidf,
                    balanced=args.balanced)
    test_data, test_labels = test.X, test.y

    apx = get_appendix(args.include_tfidf, args.balanced)
    if args.model.lower() == "simple-ff":
        preds = test_pytorch(test_data, test_labels,
                             load_model_path=f"models/simple-ff{apx}.torch",
                             predictions_file=f"preds/simple-ff-preds{apx}.txt"
                             )
    elif args.model.lower() == "bi-lstm":
        model = BiLSTM(load_model_path="models/bilstm.keras",
                       tokenizer_path='models/bilstm-tokenizer.json')
        preds = model.test(test_data, y_test=test_labels)
    elif args.model.lower() == "logreg":
        model = LogisticRegression(load_model_path=f"models/logreg{apx}.pkl")
        preds = model.test(test_data, test_labels,
                           save_predictions_path=f"preds/logreg-preds{apx}.txt")
    elif args.model.lower() == "majority-vote":
        model = MajorityVote(load_model_path="models/majority-class.txt")
        preds = model.test(test_labels)
    elif args.model.lower() == "bert":
        model = Bert(load_model_path="models/bert.pkl")
        preds = model.test(test_data, test_labels, 
                    save_predictions_path="preds/bert-preds.txt")
    elif args.model.lower() == "svm":
        model = SVM(load_model_path=f"models/svm{apx}.sav")
        preds = model.test(test_data, save_predictions_path=f"preds/svm-preds{apx}.txt")
    else:
        raise Exception("Unknown model type passed in!")

    metrics = classification_report(test_labels, preds, output_dict=True)
    pprint(metrics)
    with open(f"scores/{args.model.lower()}{apx}.json", "w") as fout:
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
