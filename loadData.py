# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
import re

# Reading the csv file into a pandas DataFrame
newsData = pd.read_csv('news_rates.csv', encoding = "ISO-8859-1")

tokenizer = TreebankWordTokenizer()
stemmer = WordNetLemmatizer()
tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 2))

def data_cleaner(text):
	letters_only = re.sub("[^a-zA-Z]", " ", text)
	lower_case = letters_only.lower()
	tokens = tokenizer.tokenize(lower_case)
	base_words = [stemmer.lemmatize(token) for token in tokens]
	return (" ".join(base_words)).strip()

testing = newsData.text[:10]
test_result = []
for row in testing:
	test_result.append(data_cleaner(row))
	
features = tfidf.fit_transform(test_result)

featuresDf = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)

print(featuresDf.head())