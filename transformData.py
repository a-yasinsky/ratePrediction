# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
newsData = pd.read_csv('newsData.csv', encoding = "ISO-8859-1", parse_dates = ['Date'])
cols = ["Date", "Rate"]
ratesData = pd.read_csv('rates.csv', encoding = "ISO-8859-1", header = None, 
						names = cols, delimiter = ";", parse_dates = ['Date'])

newsData.insert(newsData.shape[1],"Rate", 0)
newsData.drop(columns = ['name','sentiment'],axis=1,inplace=True)
newsData.drop(newsData.columns[0],axis=1,inplace=True)

#print(ratesData[0][0])
print(newsData.head())
print(ratesData.head())