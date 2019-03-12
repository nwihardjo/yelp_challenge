The csv's has already been preprocessed to speed up the pre-processing while running the model.

- column `polarity` and `subjectivity` are the sentiment analysis value from `textblob` library
- column `business_stars` and `business_rcount` are the overall star and total review for each business id. It is extracted from `business.json` file downloaded from yelp directly
- column `s0` to `s9` are the sentiment analysis values from `pycorenlp` library (a python wrapper of `stanfordcorenlp` library. Each column (i.e. `s0`, `s1`) coressponds to the sentiment analysis values for each sentence in the `text` value.

These columns are used in the model, extracted in the `read_data` function. I calculated the mean of the sentiment analysis values from `pycorenlp` eventually.
