# Importing pandas and numpy and others
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

csv = 'clean_news.csv'
news_df = pd.read_csv(csv,index_col=0)
print(news_df.head())

x = news_df.text
y = news_df.target
SEED = 2000
# Split the data into training/testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.02, random_state=SEED)
print(len(x_train), len(x_test))
print(len(y_train), len(y_test))

def tfidf_features(x_train, x_test):
    
	tfidf = TfidfVectorizer(min_df=2, max_df=0.5, ngram_range=(1, 3))
	#tfidf features form clean text
	train_features = tfidf.fit_transform(x_train)
	test_features = tfidf.transform(x_test)
	return train_features, test_features
	'''
	featuresDf = pd.DataFrame(
		features.todense(),
		columns=tfidf.get_feature_names()
	)
	print(featuresDf.head())
	'''
x_train_features, x_test_features = tfidf_features(x_train, x_test)
# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(x_train_features, y_train)

# Make predictions using the testing set
y_pred = regr.predict(x_test_features)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))
