import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.options.display.float_format = '{:.2f}'.format
sns.set(rc={'figure.figsize':(20, 20)})
In [3]:

# import sys
# print("Python version: {}". format(sys.version))
#
# import pandas as pd
# print("pandas version: {}". format(pd.__version__))
#
# import matplotlib
# print("matplotlib version: {}". format(matplotlib.__version__))
#
# import numpy as np
# print("NumPy version: {}". format(np.__version__))
#
# import scipy as sp
# print("SciPy version: {}". format(sp.__version__))
#
# import IPython
# from IPython import display
# print("IPython version: {}". format(IPython.__version__))
#
# import sklearn
# print("scikit-learn version: {}". format(sklearn.__version__))
#
# import keras
# print("keras version: {}".format(keras.__version__))
#
# import tensorflow as tf
# print("tensorflow version: {}".format(tf.__version__))

df = pd.read_csv('data/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv')

df = pd.read_csv('data/stock-time-series-20050101-to-20171231/all_stocks_2006-01-01_to_2018-01-01.csv', parse_dates=['Date'])

df.Date = pd.to_datetime(df.Date)

df.isnull().sum()

rng = pd.date_range(start='2006-01-01', end='2018-01-01', freq='B')
rng[~rng.isin(df.Date.unique())]


df.groupby('Name').count().sort_values('Date', ascending=False)['Date']

gdf = df[df.Name == 'AABA']
cdf = df[df.Name == 'CAT']

cdf[~cdf.Date.isin(gdf.Date)]


df.Name.unique().size

df.groupby('Date').Name.unique().apply(len)
Out[17]:


df.set_index('Date', inplace=True)

#Backfill `Open` column
values = np.where(df['2017-07-31']['Open'].isnull(), df['2017-07-28']['Open'], df['2017-07-31']['Open'])
df['2017-07-31']= df['2017-07-31'].assign(Open=values.tolist())

values = np.where(df['2017-07-31']['Close'].isnull(), df['2017-07-28']['Close'], df['2017-07-31']['Close'])
df['2017-07-31']= df['2017-07-31'].assign(Close=values.tolist())

values = np.where(df['2017-07-31']['High'].isnull(), df['2017-07-28']['High'], df['2017-07-31']['High'])
df['2017-07-31']= df['2017-07-31'].assign(High=values.tolist())

values = np.where(df['2017-07-31']['Low'].isnull(), df['2017-07-28']['Low'], df['2017-07-31']['Low'])
df['2017-07-31']= df['2017-07-31'].assign(Low=values.tolist())

df.reset_index(inplace=True)

df[df.Date == '2017-07-31']

missing_data_stocks = ['CSCO','AMZN','INTC','AAPL','MSFT','MRK','GOOGL', 'AABA']



columns = df.columns.values
