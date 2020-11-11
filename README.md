# ML-Fall2020-Final-Project

## Creating your credentials file

1. Create a file `scripts/CREDS.py`. 
2. Copy the contents of `scripts/CREDS_example.py` into `scripts/CREDS.py`.
3. Fill out `scripts/CREDS.py` with your Twitter API credentials. 


## Collecting Data

To query the twitter api, in the `scripts` directory, run 
```
$ python access_twitter_api.py [N] # Number of tweets to collect \
    --query [query] # Optional parameter for search query, Default: "#Election2020" \
    --datadir [/path/to/data] # Optional path to data directory, Default: "../data"
```

The script will save the data to the following path format: `{datadir}/{query}_{%d-%m-%Y_%H:%M:%S}.tsv`