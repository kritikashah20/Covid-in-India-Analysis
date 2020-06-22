# Covid in India

# Basic packages
import warnings
warnings.filterwarnings('ignore')
import pandas as pd    # read the dataset
import numpy as np    # numerical python or array for matrix multiplication
import matplotlib.pyplot as plt    # to plot the graph
import seaborn as sns    # graphical representation

# import dataset
covid = pd.read_csv('covid_19_india.csv')
print(covid.head())
print(covid.tail())
print(covid.shape)

# to check if null is present or not
covid.isnull()
covid.isnull().sum()

# if null value present then
covid.dropna(inplace = True)

# checking the info whether correct or not
print(covid.info())

# plot graph for a state 
df = covid.loc[(covid['State/UnionTerritory'] == 'Kerala')]
df.head()
df.shape

# plot graph
sns.countplot(x = 'ConfirmedIndianNational', data = df)

import plotly.offline as py
import plotly.graph_objs as go

cured_chart = go.Scatter(x = df['Date'], y = df['Cured'], name = 'Cured Rate')
death_chart = go.Scatter(x = df['Date'], y = df['Deaths'], name = 'Death Rate')
py.iplot([cured_chart, death_chart])

# SVM
from sklearn.svm import LinearSVR
dfc = df[['Confirmed']]
dfc = dfc.values    # convert to array
print(dfc)

# Train and Test
train_size = int(len(dfc) * 0.80)
test_size = len(dfc) - train_size

train, test = dfc[0:train_size,:], dfc[train_size:len(dfc),:]
train.shape
test.shape

# for prediction result
def create(dataset, look_back = 1):    # check yesterdays and tdays both data if lookback is 2 then todays data then skip 2 days
	data_x, data_y = [],[]
	for i in range(len(dataset)-look_back-1):
		a = dataset[i:(i+look_back), 0]
		data_x.append(a)
		data_y.append(dataset[i+look_back, 0])
	return np.array(data_x), np.array(data_y)

look_back = 2
train_x, train_y = create(train, look_back=look_back)
test_x, test_y = create(test, look_back=look_back)

print(train_x)
print(test_x)

model = LinearSVR()
model.fit(train_x, train_y)

predict1 = model.predict(test_x)

plt.plot(test_y, color='blue', label='Actual Values')
plt.plot(predict1, color='brown', label='Predicted Values')
plt.ylabel('meantemp')
plt.legend()

df1 = pd.DataFrame({'Actual' : test_y.flatten(), 'Predicted' : predict1.flatten()})
print(df1)

df1.plot(kind = 'bar', figsize = (16, 10))










