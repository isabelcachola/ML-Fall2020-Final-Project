""" Main function, which trains our model and makes predictions with it. """
import argparse as ap
import logging
from sklearn.metrics import classification_report
from data import load
from models import FeedForward, BiLSTM, LogisticRegression, MajorityVote, SVM
from utils import train_pytorch, test_pytorch, get_appendix
from pprint import pprint
from imblearn.over_sampling import SMOTE
import json
import pandas as pd


def balance_train_data(x_train, y_train):
    """
    This function balances the data by up-sampling the minority class
    :param x_train: the train x data
    :param y_train: the train labels
    :return: the up-sampled x (in dataframe format) and y (in numpy array format) data
    """

    smote = SMOTE()
    # fit predictor and target variable
    x_smote, y_smote = smote.fit_resample(x_train, y_train)
    return pd.DataFrame(x_smote, columns=x_train.columns), y_smote


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

    # hyper parameters
    p.add_argument("--batch-size", type=int, default=36)

    # simple-ff hparams
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--train-steps", type=int, default=500)
    p.add_argument("--ff-hunits", type=int, default=128)

    # bi-lstm hparams
    p.add_argument("--num-epochs", type=int, default=5)

    return p.parse_args()


def train(args):
    """
    This function trains the models
    :param args: the command line arguments defining the desired actions
    """

    # load data
    train_data_all, dev_data_all, _ = load(args.data_dir, cachedir=args.cachedir,
                                           override_cache=args.override_cache,
                                           text_only=(args.model.lower() == "bi-lstm"),
                                           include_tfidf=args.include_tfidf)
    train_data, train_labels = train_data_all.X, train_data_all.y
    dev_data, dev_labels = dev_data_all.X, dev_data_all.y

    # Check if should balance data
    if args.balanced:
        train_data_balanced, train_labels_balanced = balance_train_data(train_data, train_labels)

    # Build model
    apx = get_appendix(args.include_tfidf, args.balanced)
    if args.model.lower() == "simple-ff":
        model = FeedForward(args.ff_hunits, train.X.shape[1])
        train_pytorch(args, model,
                      train_data, train_labels,
                      dev_data, dev_labels,
                      save_model_path=f"models/simple-ff{apx}.torch")
    elif args.model.lower() == "bi-lstm":
        model = BiLSTM(epochs=args.num_epochs, batch_size=args.batch_size)
        model.train(train_data, train_labels, dev_data, dev_labels)
    elif args.model.lower() == "logreg":
        model = LogisticRegression()
        if args.balanced:
            model.train(train_data_balanced, train_labels_balanced, dev_data, dev_labels,
                        save_model_path=f"models/logreg{apx}.pkl")
        else:
            model.train(train_data, train_labels, dev_data, dev_labels,
                        save_model_path=f"models/logreg{apx}.pkl")
    elif args.model.lower() == "majority-vote":
        model = MajorityVote()
        model.train(train_labels, dev_labels)
    elif args.model.lower() == "svm":
        model = SVM()
        if args.balanced:
            model.train(train_data_balanced, train_labels_balanced,
                        save_model_path=f"models/svm{apx}.sav")
        else:
            model.train(train_data, train_labels, save_model_path=f"models/svm{apx}.sav")
    else:
        raise Exception("Unknown model type passed in!")


def test(args):
    """
    This function tests our models
    :param args: the command line arguments with the desired actions
    """
    _, _, test_data_all = load(args.data_dir, cachedir=args.cachedir,
                               override_cache=args.override_cache,
                               text_only=(args.model.lower() == "bi-lstm"),
                               include_tfidf=args.include_tfidf)
    test_data, test_labels = test_data_all.X, test_data_all.y

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
