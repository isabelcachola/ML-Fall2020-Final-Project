# ML-Fall2020-Final-Project

## Creating your credentials file

You only need to do this if you need to access the Twitter API
1. Create a file `scripts/CREDS.py`. 
2. Copy the contents of `scripts/CREDS_example.py` into `scripts/CREDS.py`.
3. Fill out `scripts/CREDS.py` with your Twitter API credentials. 


## Data

### Raw Data

You only need to complete the following steps if you are making changes to the preprocessing script. Otherwise, skip to the Preprocessed Data.

0. (Only Isabel can do this step bc only she has access to the grid. She uploaded the output of this script to the Drive.) On the CLSP grid `python parse_clsp_data.py -d /path/to/mark/data/ -o /path/to/mydir/ --num_cores N`. 
1. Download the raw data from the Google Drive. It's called `raw_twitter_data.tar.gz`. This is a compressed version of the data from the CLSP grid.
2. Unzip by running `tar -xf raw_twitter_data.tar.gz` To learn more about `.tar.gz` files, see [here.](https://linuxize.com/post/how-to-extract-unzip-tar-gz-file/)
3. To run the preprocessing script, run
 `$ python preprocess_tweets.py --datadir /path/to/raw/data/ --outdir /path/to/data/`
 4. Upload the updated `all_data_preprocess.tsv` file to the drive.

 ### Preprocessed Data

Download the `all_data_preprocess.tsv` file from the drive. It's a tab separated file. It contains the following columns:
```
id: Tweet ID
text: Tweet text, cleaned
author_id: Tweet author id
retweet_count: int
reply_count: int
like_count: int
quote_count: int
mentions: list of mentions
hashtags: string of hashtags
label: Bindary label to predict
hashtags_tfidf: Precomputed hashtag tfidf scores
sentiment_score_pos: Precomputed sentiment score positive
sentiment_score_neu: Precomputed sentiment score neutrial
sentiment_score_neg: Precomputed sentiment score negative
sentiment_score_comp: Precomputed sentiment score composed
```