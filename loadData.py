# Importing pandas and numpy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import TreebankWordTokenizer
import re

# Reading the csv file into a pandas DataFrame
newsData = pd.read_csv('news_rates.csv', encoding = "ISO-8859-1")

tokenizer = TreebankWordTokenizer()

def data_cleaner(text):
	letters_only = re.sub("[^a-zA-Z]", " ", text)
	lower_case = letters_only.lower()
	words = tokenizer.tokenize(lower_case)
	return (" ".join(words)).strip()

testing = newsData.text[:10]
test_result = []
for row in testing:
	test_result.append(data_cleaner(row))
	
print(test_result)