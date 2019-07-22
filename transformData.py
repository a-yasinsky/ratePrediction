# Importing pandas and numpy
import pandas as pd
import numpy as np

# Reading the csv file into a pandas DataFrame
newsData = pd.read_csv('newsData.csv', encoding = "ISO-8859-1", parse_dates = ['Date'])
cols = ["Date", "Rate"]
ratesData = pd.read_csv('rates.csv', encoding = "ISO-8859-1", header = None, 
						names = cols, delimiter = ";", parse_dates = ['Date'])

newsData = newsData[newsData["name"] == "Russia"].reset_index()
newsData.insert(newsData.shape[1],"Rate", 0.0)
newsData.drop(columns = ['name','sentiment'],axis=1,inplace=True)
newsData.drop(newsData.columns[0],axis=1,inplace=True)

ratesDict = {}
for i in range(len(newsData)):
	newsDate = newsData.loc[i, "Date"]
	if newsDate in ratesDict:
		rateForDate = ratesDict[newsDate]
	else:
		rateForDate = ratesData[ratesData["Date"] == newsDate].reset_index()
		rateForDate = rateForDate["Rate"][0]
		ratesDict[newsDate] = rateForDate
	newsData.at[i, "Rate"] = rateForDate
newsData.to_csv('news_rates_ru.csv',encoding='utf-8')
csv = 'news_rates_ru.csv'
my_df = pd.read_csv(csv,index_col=0)
print(my_df.head())