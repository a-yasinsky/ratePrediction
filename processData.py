# Importing pandas and numpy and others
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 3))

csv = 'clean_news.csv'
my_df = pd.read_csv(csv,index_col=0)
print(my_df.head())

'''
#tfidf features form clean text
features = tfidf.fit_transform(test_result)

featuresDf = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)

print(featuresDf.head())
'''