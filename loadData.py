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

# visualize tokens frequency
clean_df = pd.DataFrame(test_result, columns=['text'])
print(clean_df.head())

from sklearn.feature_extraction.text import CountVectorizer
cvec = CountVectorizer()
cvec.fit(clean_df.text)
print(len(cvec.get_feature_names()))

doc_matrix = cvec.transform(clean_df.text)
tf = np.sum(doc_matrix,axis=0)
pos = np.squeeze(np.asarray(tf))
term_freq_df = pd.DataFrame([pos],columns=cvec.get_feature_names()).transpose()
print(term_freq_df[0][:10])

y_pos = np.arange(50)
plt.figure(figsize=(12,10))
plt.bar(y_pos, term_freq_df[0][:50], align='center', alpha=0.5)
plt.xticks(y_pos, term_freq_df[0][:50].index,rotation='vertical')
plt.ylabel('Frequency')
plt.xlabel('Top 50 negative tokens')
plt.title('Top 50 tokens in negative tweets')
plt.show()
'''
features = tfidf.fit_transform(test_result)

featuresDf = pd.DataFrame(
    features.todense(),
    columns=tfidf.get_feature_names()
)

print(featuresDf.head())
'''