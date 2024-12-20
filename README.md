# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas and matplotlib.pyplot
2. Read the dataset and transform it
3. Import KMeans and fit the data in the model
4. Plot the Cluster graph

## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: GOKUL S
RegisterNumber:  24004336
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
data = pd.read_csv("Mall_Customers.csv")
print(data.head())
print(data.info())
print(data.isnull().sum())
wcss = []
import os
os.environ["OMP_NUM_THREADS"] = "1"
import warnings
warnings.filterwarnings("ignore", message="KMeans is known to have a memory leak")
warnings.filterwarnings("ignore", category=FutureWarning)
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init="k-means++")
    kmeans.fit(data.iloc[:, 3:])
    wcss.append(kmeans.inertia_)
plt.plot(range(1, 11), wcss)
plt.xlabel("No. of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()
km = KMeans(n_clusters=5)
km.fit(data.iloc[:, 3:])
y_pred = km.predict(data.iloc[:, 3:])
data["cluster"] = y_pred
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]
plt.scatter(df0["Annual Income (k$)"], df0["Spending Score (1-100)"], c="red", label="Cluster 0")
plt.scatter(df1["Annual Income (k$)"], df1["Spending Score (1-100)"], c="blue", label="Cluster 1")
plt.scatter(df2["Annual Income (k$)"], df2["Spending Score (1-100)"], c="green", label="Cluster 2")
plt.scatter(df3["Annual Income (k$)"], df3["Spending Score (1-100)"], c="purple", label="Cluster 3")
plt.scatter(df4["Annual Income (k$)"], df4["Spending Score (1-100)"], c="magenta", label="Cluster 4")
plt.legend()
plt.title("Customer Segments")
plt.show()
*/
```

## Output:
![Screenshot 2024-12-20 211252](https://github.com/user-attachments/assets/b001ef99-1d06-4246-b7ae-1ef7355730da)
![Screenshot 2024-12-20 211322](https://github.com/user-attachments/assets/d346a3a3-e19d-4295-b1c7-10b4c791ed1c)
![Screenshot 2024-12-20 211336](https://github.com/user-attachments/assets/c11188fc-8bcd-402b-afe0-02903e2383da)



## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
