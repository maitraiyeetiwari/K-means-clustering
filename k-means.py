#from urllib import request
#url='https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%204/data/Cust_Segmentation.csv'
#request.urlretrieve(url,'cust_segmentation.csv')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing

df=pd.read_csv('cust_segmentation.csv')
df.head()

#table has a column address which has categorical values. Drop this column. 


df=df.drop('Address',axis=1)

#preprocessing

x=np.asanyarray(df[['Age', 'Edu', 'Years Employed', 'Income', 'Card Debt','Other Debt', 'Defaulted', 'DebtIncomeRatio']])
x[:5]
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x)
x 

from sklearn.cluster import KMeans
clusterNum = 3
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 12)

x = np.nan_to_num(x)
k_means.fit(x)
labels = k_means.labels_
print('labels=',labels)

df["Labels"] = labels
df.columns
df.head()

#check the centroid values
df.groupby('Labels').mean()

#plotting
label_0=df.loc[df['Labels'] == 0]
label_1=df.loc[df['Labels'] == 1]
label_2=df.loc[df['Labels'] == 2]
#label_3=df.loc[df['Labels'] == 3]

plt.scatter(label_0['Age'],label_0['Income'],color='red',s=np.pi*(label_0['Edu'])**2,label='cluster 1')
plt.scatter(label_1['Age'],label_1['Income'],color='green',s=np.pi*(label_1['Edu'])**2,label='cluster 2')
plt.scatter(label_2['Age'],label_2['Income'],color='blue',s=np.pi*(label_2['Edu'])**2,label='cluster 3')
#plt.scatter(label_3['Age'],label_3['Income'],color='yellow',s=np.pi*(label_3['Edu'])**2,label='cluster 4')

plt.xlabel('Age')
plt.ylabel('Income')
plt.legend()
plt.title('customer clustering based on education,age and income')
plt.savefig('customer-clustering.png',dpi=300)
#plt.show()










