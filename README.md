# ML-Fall2020-Final-Project

# Data

## Creating your credentials file

You only need to do this if you need to access the Twitter API
1. Create a file `scripts/CREDS.py`. 
2. Copy the contents of `scripts/CREDS_example.py` into `scripts/CREDS.py`.
3. Fill out `scripts/CREDS.py` with your Twitter API credentials. 

## Raw Data

You only need to complete the following steps if you are making changes to the preprocessing script. Otherwise, skip to the Preprocessed Data.

0. (Only Isabel can do this step bc only she has access to the grid. She uploaded the output of this script to the Drive.) On the CLSP grid `python parse_clsp_data.py -d /path/to/mark/data/ -o /path/to/mydir/ --num_cores N`. 
1. Download the raw data from the Google Drive. It's called `raw_twitter_data.tar.gz`. This is a compressed version of the data from the CLSP grid.
2. Unzip by running `tar -xf raw_twitter_data.tar.gz` To learn more about `.tar.gz` files, see [here.](https://linuxize.com/post/how-to-extract-unzip-tar-gz-file/)
3. To run the preprocessing script, run
 `$ python preprocess_tweets.py --datadir /path/to/raw/data/ --outdir /path/to/data/`
 4. Upload the updated `all_data_preprocess.tsv` file to the drive.

The files `data/{train/dev/test}-ids.txt` contain tweet ids with for each split of the dataset. **MAKE SURE TO USE THESE SPLITS WHEN TRAINING/TUNING/TESTING.** 

## Preprocessed Data

Download the `all_data_preprocess.tsv` file from the drive. It's a tab separated file. 

**You still have to featurize this data, depending on your model choice.**

It contains the following columns:
```
id: Tweet ID
text: Tweet text
processed_text: Tweet text, cleaned
author_id: Tweet author id
retweet_count: int
reply_count: int
like_count: int
quote_count: int
author_followers: int
mentions: list of mentions
mentions_count: int
hashtags: string of hashtags
label: Bindary label to predict
hashtags_tfidf: Precomputed hashtag tfidf scores
sentiment_score_pos: Precomputed sentiment score positive
sentiment_score_neu: Precomputed sentiment score neutrial
sentiment_score_neg: Precomputed sentiment score negative
sentiment_score_comp: Precomputed sentiment score composed
```

# Training
To train a model, run the following command:
```
$ python main.py train {logreg|bi-lstm|simple-ff} {optional parameters}
```
The script will cache the featurized data in `data/`. If you are making changes to the featurization, use the flag `--override-cache`.

The script saves the modeling weights to `models/`.

# Testing

To test a model, run the following command:
To train a model, run the following command:
```
$ python main.py predict {logreg|bi-lstm|simple-ff} {optional parameters}
```

The script saves predictions to `preds/` and testing metrics to `scores/`


