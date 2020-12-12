import pandas as pd
from imblearn.over_sampling import SMOTE
import argparse
import os


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadirx', '-dx', dest='datadirx', type=str,
                        help='File from which to get the X train data',
                        default="../data/raw_data/xtrain.csv")

    parser.add_argument('--datadiry', '-dy', dest='datadiry', type=str,
                        help='File from which to get the y train data',
                        default="../data/raw_data/ytrain.csv")

    parser.add_argument('--outdir', '-o', dest='outdir', type=str,
                        help='Directory to store the balanced train data',
                        default="../data/balanced")

    args = parser.parse_args()

    x_train = pd.read_csv(args.datadirX)
    y_train = pd.read_csv(args.datadiry)

    x_upsampled, y_upsampled = balance_train_data(x_train, y_train)
    x_upsampled["label"] = y_upsampled

    outpath = os.path.join(args.outdir, 'train_upsampled.csv')
    x_upsampled.to_csv(outpath, index=False)
