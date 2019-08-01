import pandas as pd  
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models.doc2vec import LabeledSentence
import multiprocessing
from gensim.models import Doc2Vec
from tqdm import tqdm
from sklearn import utils
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

def labelize_news(news,label):
    result = []
    prefix = label
    for i, t in zip(news.index, news):
        result.append(LabeledSentence(t.split(), [prefix + '_%s' % i]))
    return result


x_w2v = labelize_news(x, 'all')

cores = multiprocessing.cpu_count()

VECTOR_SIZE = 100

model_dbow = Doc2Vec(dm=0, vector_size=VECTOR_SIZE, negative=5, min_count=2, workers=cores, alpha=0.065, min_alpha=0.065)
model_dbow.build_vocab([x for x in tqdm(x_w2v)])

for epoch in range(10):
    model_dbow.train(utils.shuffle([x for x in tqdm(x_w2v)]), total_examples=len(x_w2v), epochs=1)
    model_dbow.alpha -= 0.002
    model_dbow.min_alpha = model_dbow.alpha
	
def get_vectors(model, corpus, size):
    vecs = np.zeros((len(corpus), size))
    n = 0
    for i in corpus.index:
        prefix = 'all_' + str(i)
        vecs[n] = model.docvecs[prefix]
        n += 1
    return vecs

train_vecs_dbow = get_vectors(model_dbow, x_train, VECTOR_SIZE)
test_vecs_dbow = get_vectors(model_dbow, x_test, VECTOR_SIZE)

# Create linear regression object
regr = LinearRegression()

# Train the model using the training sets
regr.fit(train_vecs_dbow, y_train)

# Make predictions using the testing set
y_pred = regr.predict(test_vecs_dbow)

print("Mean squared error: %.2f"
      % mean_squared_error(y_test, y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(y_test, y_pred))