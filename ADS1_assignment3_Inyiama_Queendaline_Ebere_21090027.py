# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 02:40:09 2023

@author: EbereInyiama
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt

import err_ranges as err
import map

def worldbank(dataset, indicator):
    
    """This is function to read the dataset csv file into a pandas dataframe"""
    
    df_dataset = pd.read_csv(dataset, skiprows=4)
    df_dataset = df_dataset.drop(['Country Code', 'Indicator Code'], axis = 1)
    df_dataset = df_dataset[df_dataset['Indicator Name'] == indicator]
   
    return df_dataset 

# creating variables for dataset and indicator
dataset = 'API_19_DS2_en_csv_v2_4773766.csv'
indicator = 'Population growth (annual %)'

df_clustering = worldbank(dataset, indicator).T
df_clustering = df_clustering.rename(columns=df_clustering.iloc[0])
df_clustering = df_clustering.drop(index=df_clustering.index[:2], axis=0)
df_clustering = df_clustering.fillna(0)
print()
print(df_clustering)

# selecting a few countries to find interesting clustering
df_clustering = df_clustering[['Nigeria', 'China', 'India', 'Canada', 'Brazil', 'Germany', 'Australia', 
                                  'Morocco', 'Singapore', 'Ireland', 'South Africa', 'United States']]

# heatmap
map.heat_corr(df_clustering, 9)

pd.plotting.scatter_matrix(df_clustering, figsize=(9.0, 9.0))
plt.tight_layout()    # this helps to avoid overlap of labels
plt.show()

# A function to normalize dataset
def norm(df, col1, col2):
    """This function returns normalised values of [0,1] on the columns of the dataframe
    """
    df[col1] = (df[col1] - df[col1].min())/(df[col1].max() -df[col1].min())
    df[col2] = (df[col2] - df[col2].min())/(df[col2].max() -df[col2].min())
    
    return df

# extract columns for fitting and apply normalization only on the extracted columns by creating a copy.
df_fit = df_clustering[["Australia", "South Africa"]].copy()
df_fit = norm(df_fit, 'Australia', 'South Africa')

#print()
print(df_fit)

# A scatter plot of the two countries
df_fit.plot('Australia', 'South Africa', kind='scatter')

#To set up kmeans and fit
for ic in range(2, 7):

    kmeans = KMeans(n_clusters=ic)
    kmeans.fit(df_fit)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(df_fit, labels))
    
# Using elbow method to determine the number of clusters

numClusters = [1,2,3,4] #to find the best number of clusters
SSE = []
for k in numClusters:
    k_means = KMeans(n_clusters=k)
    k_means.fit(df_fit)
    SSE.append(k_means.inertia_)

plt.plot(numClusters, SSE)
plt.title('Elbow method')
plt.xlabel('Number of Clusters')
plt.ylabel('SSE')

# A plot for 3 clusters
kmeans = KMeans(n_clusters=3)
kmeans.fit(df_fit)     

# extracting labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(5.0, 5.0))

plt.scatter(df_fit["Australia"], df_fit["South Africa"], c=labels, cmap="Accent")


# showing cluster centres
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("Australia")
plt.ylabel("South Africa")
plt.title("3 clusters")
plt.show()

#-----------------------
# A plot for 2 clusters
kmeans = KMeans(n_clusters=2)
kmeans.fit(df_fit)     

# extracting labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(figsize=(5.0, 5.0))

plt.scatter(df_fit["Australia"], df_fit["South Africa"], c=labels, cmap="Accent")


# show cluster centres
for ic in range(2):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("Australia")
plt.ylabel("South Africa")
plt.title("2 clusters")
plt.show()