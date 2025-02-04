# Alenmathew.Capstoneproject
# Data cleaning
import pandas as pd
df = pd.read_csv("TiresDataFile_DSCI490_2024AY.csv") df.head()
print(df.shape)
df.info()
## percentage of missing values in columns

(df.isnull().sum()/len(df)) * 100
## columns with missing values
df.columns[df.isnull().sum()>0]
## total count of missing values in columns
df[df.columns[df.isnull().sum()>0]].isnull().sum()
## dropping redundant and missing columns
df = df.drop(["Additional vehicle(s)" , "Account No."], axis = 1)
Imputing values
Categorical Features
## Household income
df['Household income'] = df['Household income'].fillna(df['Household income'].mode()[0])
## Vehicle types
df['Primary Vehicle type'] = df['Primary Vehicle type'].fillna(df['Primary Vehicle type'].mode()[0]) df['2nd Vehicle type'] = df['2nd Vehicle type'].fillna('Not Available')
df['3rd Vehicle type'] = df['3rd Vehicle type'].fillna('Not Available')
df['4th Vehicle type'] = df['4th Vehicle type'].fillna('Not Available')
## Highest quality of rim purchased

df['Highest quality of rim purchased in the last 5 years'] = df['Highest quality of rim purchased in the last 5 years'].fillna(df['Highest quality of rim purchased in the last 5 years'].mode()[0])
## Discount code obtained

df.rename(columns={'Discount code obtained ': 'Discount code obtained'}, inplace=True)

df['Discount code obtained'] = df['Discount code obtained'].fillna(df['Discount code obtained'].mode()[0]) df['Discount code obtained'].replace({'0': df['Discount code obtained'].mode()[0]}, inplace=True)
## Discount code
df['Discount code'].replace({'NA': '0', '#NA' : '0'}, inplace=True) df['Discount code'] = df['Discount code'].fillna(df['Discount code'].mode()[0])
## Promo responses as 'N'

df['Responded\nMarch Promo (summer tires)'].fillna('N', inplace = True) df['Responded\nMarch promo (all seasons)'].fillna('N', inplace = True) df['Responded\nAugust Promo (all-season tires)'].fillna('N', inplace = True) df['Responded\nOctober Promo (winter tires)'].fillna('N', inplace = True) df['Responded\nNovember promo (winter tires)'].fillna('N', inplace = True)
Numerical feature
## Birth Year
df['Cust. Year_Birth'].fillna(df['Cust. Year_Birth'].median(), inplace = True)
## KM per year
df['KM per year'].replace({0:df['KM per year'].median()}, inplace = True)
## Sets of summer tires

df['Summer tires \n(# sets purchased)'].fillna(df['Summer tires \n(# sets purchased)'].median(), inplace = True)
## Converting dates to datetime

df['Last Date of purchase summer'] = pd.to_datetime(df['Last Date of purchase summer'], format = "%m/%d/%Y") df['Last Date of purchase winter'] = pd.to_datetime(df['Last Date of purchase winter'], format = "%m/%d/%Y") df['Last Date of purchase all seasons'] = pd.to_datetime(df['Last Date of purchase all seasons'], format = "%m/%d/%Y")
df.to_excel('tire_output_file.xlsx', index=False)

# Exploratory data analysis
import pandas as pd
df = pd.read_excel("tire_output_file.xlsx")
df.head()
print(df.shape) 
print(df.dtypes)
df.isnull().sum()
df['Household income'] = df['Household income'].map(
{'< 32000':0 , '32000 - 55000':1, '55000 - 75000':2,'75000 - 99000':3,
'99000 - 140000':4,'140000 - 200000':5,'> 200000':6})

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np

plt.figure(figsize=(18,8)) sns.countplot(data=df, x='Household income')

plt.xticks(np.arange(7),
['Under \$32000' , '\$32000 to\n\$55000', '\$55001 to\n \$75000','\$75001 to\n \$99000', '\$99001 to\n \$140000','\$140001 to\n \$200000','\$200001 & Over'],
fontsize=13)
#plt.title("Distribution of Customer Based on Income", fontsize = 18)
plt.vlines(2.5, ymin=0, ymax=450, linestyles='dashed',label='$75000, Median Household Income') plt.legend(fontsize=15)
plt.xlabel("") plt.ylabel("", fontsize=15) plt.xticks(fontsize=12) plt.yticks(fontsize=12)
plt.title("Distribution of Customers Based on Income", fontsize = 18) plt.show()
df.columns
seasons = df[['Summer tires \n(# sets purchased)',
'Winter tires \n(# sets purchased)', 'All seasons\n(# sets purchased)', 'Last Date of purchase summer', 'Last Date of purchase winter',
'Last Date of purchase all seasons']]

seasons.shape
seasons.dtypes
summer = pd.DataFrame()
summer['month'] = seasons['Last Date of purchase summer']
summer['Summer tire sets'] = seasons['Summer tires \n(# sets purchased)']
winter = pd.DataFrame()
winter['month'] = seasons['Last Date of purchase winter']
winter['Winter tire sets'] = seasons['Winter tires \n(# sets purchased)']
All_season = pd.DataFrame()
All_season['month'] = seasons['Last Date of purchase all seasons'] 
All_season['Winter tire sets'] = seasons['All seasons\n(# sets purchased)']
print(summer.month.min()) print(winter.month.min()) 
print(All_season.month.min())
summer.rename(columns = {'Summer tire sets': "Sets Sold"}, inplace = True)

winter.rename(columns = {'Winter tire sets' : "Sets Sold"}, inplace = True)
All_season.rename(columns = {'Winter tire sets' : "Sets Sold"}, inplace = True)

summer['Season'] = "Summer" 
winter['Season'] = "Winter" 
All_season['Season'] = "All Season"
sales = pd.concat([summer, winter, All_season])
sales.shape
sales.to_excel('sales.xlsx', index=False)

# Sales and customer segmentation
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
df = pd.read_excel("sales.xlsx") 
df.head()
df = df.iloc[:, 0:2] 
df.head()
df['year'] = df['month'].dt.year
df.head()
series = df.groupby(['year'])['Sets Sold'].sum() 
series
series = pd.DataFrame(series) 
series
series.index
series.index = pd.to_datetime(series.index, format='%Y').to_period('Y')
series.index
series
series = series['2019': '2022'] series
model = ExponentialSmoothing(endog = series['Sets Sold'], trend = 'add').fit()
prediction = model.forecast(steps = 3 )
series['Sets Sold'].plot() prediction.plot()
new_df = pd.read_excel("tire_output_file.xlsx")
new_df['Age'] = 2024 - new_df['Cust. Year_Birth']
## Clustering
Cluster_df = pd.read_csv("Tire_set.csv") Cluster_df.head()
Cluster_df['Age'] = new_df['Age']
Cluster_df.columns

from sklearn.cluster import KMeans
import seaborn as sns wss = []
for k in range(1, 10):
kmeans = KMeans(n_clusters=k, init='k-means++') kmeans.fit(Cluster_df) wss.append(kmeans.inertia_)

sns.lineplot(range(1, 10), wss, marker='o') plt.xticks(range(1, 10))
plt.xlabel('Number of Clusters') plt.ylabel('Within-Cluster Sum of Squares') plt.title('Elbow Plot for K-means Clustering') plt.show()
kmeans=KMeans(n_clusters=4, init="k-means++") 
Cluster=kmeans.fit_predict(Cluster_df) 
Cluster_df['Cluster'] = Cluster
Cluster_df.head()
Cluster_df['Cluster'].unique()
cluster_1 = Cluster_df[Cluster_df['Cluster']==0] 
cluster_2 = Cluster_df[Cluster_df['Cluster']==1] 
cluster_3 = Cluster_df[Cluster_df['Cluster']==2]
print("{:>25} {:>10} {:>10} {:>10}".format('Variable', 'Cluster 1', 'Cluster 2', 'Cluster 3'))
for i in cluster_1.columns:
print("{:>25} {:>10} {:>10} {:>10}".format(i, cluster_1[i].median(), cluster_2[i].median(), cluster_3[i].median()))
sns.scatterplot(Cluster_df['Age'],Cluster_df['Annual_Km'], hue=Cluster_df['Cluster'])
pip install nbconvert[webpdf]

