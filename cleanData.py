# Importing pandas and numpy and others
import pandas as pd
import numpy as np
from nltk.tokenize import TreebankWordTokenizer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
import re

# Reading the csv file into a pandas DataFrame
newsData = pd.read_csv('news_rates.csv', encoding = "ISO-8859-1")

#initializing tokenizer, lemmatizer, vectorizers
tokenizer = TreebankWordTokenizer()
stemmer = WordNetLemmatizer()
cvec = CountVectorizer()

#data clening and making tokens form text
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
def print_token_frequency(data_set):
	clean_df = pd.DataFrame(data_set, columns=['text'])
	cvec.fit(clean_df.text)
	print(len(cvec.get_feature_names()))
	doc_matrix = cvec.transform(clean_df.text)
	tf = np.sum(doc_matrix,axis=0)
	pos = np.squeeze(np.asarray(tf))
	term_freq_df = pd.DataFrame([pos],columns=cvec.get_feature_names()).transpose()
	print(term_freq_df.sort_values(by=[0], ascending=False)[:10])

#print_token_frequency(test_result)

texts = newsData.text
clean_texts = []
for row in texts:
	clean_texts.append(data_cleaner(row))

clean_df = pd.DataFrame(clean_texts,columns=['text'])
clean_df['target'] = newsData.Rate
print(clean_df.head())

clean_df.to_csv('clean_news.csv',encoding='utf-8')