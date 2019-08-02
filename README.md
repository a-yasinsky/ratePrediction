# Rate prediction

Rate prediction training system for learning NLP

## Dataset
* [Global news dataset](https://www.kaggle.com/dbs800/global-news-dataset/)

## Files Description
1. transformData - cleans news to contain only English news, combines with rates for dates
2. cleanData - cleans every news, tokenize and lemmatize them
3. processData - counts tfidf features from tokens, reduce dimensionality, use Linear Regression to predict rates
4. prDataD2V - trains Doc2Vec model to get features from tokens, use Linear Regression to predict rates

## Dependencies
* [pandas](https://pandas.pydata.org/)
* [NumPy](https://numpy.org/)
* [NLTK](https://www.nltk.org/)
* [scikit-learn](https://scikit-learn.org/stable/)
* [gensim](https://radimrehurek.com/gensim/)
