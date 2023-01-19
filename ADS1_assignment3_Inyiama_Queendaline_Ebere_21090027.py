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


### CLUSTERING

def worldbank(dataset, indicator):
    
    """This is function to read the dataset csv file into a pandas dataframe"""
    
    df_dataset = pd.read_csv(dataset, skiprows=4)
    df_dataset = df_dataset.drop(['Country Code', 'Indicator Code'], axis = 1)
    df_dataset = df_dataset[df_dataset['Indicator Name'] == indicator]
   
    return df_dataset 

# creating variables for dataset and indicator for clustering
dataset = 'API_19_DS2_en_csv_v2_4773766.csv'
indicator = 'Population growth (annual %)'

df_clustering = worldbank(dataset, indicator).T
df_clustering = df_clustering.rename(columns=df_clustering.iloc[0])
df_clustering = df_clustering.drop(index=df_clustering.index[:2], axis=0)
df_clustering = df_clustering.fillna(0)
print()
print(df_clustering)

# selecting a few countries to find interesting clustering
df_clustering_1 = df_clustering[['Nigeria', 'China', 'India', 'Canada', 'Brazil', 'Germany', 'Australia', 
                                  'Morocco', 'Singapore', 'Ireland', 'South Africa', 'United States']]

# heatmap
map.heat_corr(df_clustering_1, 9)

pd.plotting.scatter_matrix(df_clustering_1, figsize=(9.0, 9.0))
plt.tight_layout()    # this helps to avoid overlap of labels
plt.show()

# A function to normalize dataset for clustering
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

# A scatter plot of the two countries chosen
plt.figure()
plt.scatter(df_fit['Australia'], df_fit['South Africa'])
plt.title('Scatter Plot')
plt.xlabel('Australia')
plt.ylabel('South Africa')
plt.show()

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

# 2 clusters and 3 clusters were plotted for comparison

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

# To know the years that are similar in population annual growth for the two countries
year_kmeans = kmeans.fit_predict(df_fit)
year_kmeans

df_fit['label'] = year_kmeans
df_label_0 = df_fit.loc[df_fit['label'] == 0]

print()
print('The years in cluster 0 are \n', df_label_0.head(30)) 

df_label_1 = df_fit.loc[df_fit['label'] == 1]
print()
print('The years in cluster 1 are \n', df_label_1.head(30))
print()

### FITTING

# The fitting is done with the indicator Population Total for Nigeria

def fitting(dataset_2, indicator_2):
    
    """This is function to read the dataset csv file into a pandas dataframe
    The dataset is for fitting"""
    
    df_dataset_2 = pd.read_csv(dataset_2, skiprows=4)
    df_dataset_2 = df_dataset_2.drop(['Country Code', 'Indicator Code'], axis = 1)
    df_dataset_2 = df_dataset_2[df_dataset_2['Indicator Name'] == indicator_2]
   
    return df_dataset_2

dataset_2 = 'API_19_DS2_en_csv_v2_4773766.csv'
indicator_2 = 'Population, total'

df_fitting = fitting(dataset_2, indicator_2).T
df_fitting = df_fitting.rename(columns=df_fitting.iloc[0])
df_fitting = df_fitting.drop(index=df_fitting.index[:2], axis=0)
df_fitting = df_fitting.fillna(0)
df_fitting['Year'] = df_fitting.index
df_fitting = df_fitting[['Year', 'Nigeria']].apply(pd.to_numeric, 
                                               errors='coerce') #to set any invalid parsing as NaN
print()
print(df_fitting)

def exp_growth(t, scale, growth):
    """ This function computes exponential function 
    with scale and growth as free parameters.
    1960 is the base year.
    """
    
    f = scale * np.exp(growth * (t-1960)) 
    
    return f

# fitting the exponential growth
popt, covar = opt.curve_fit(exp_growth, df_fitting["Year"], 
                            df_fitting["Nigeria"])

print("The fit parameter: \n", popt)

# The *popt is used to pass on the fit parameters
df_fitting["pop_exp"] = exp_growth(df_fitting["Year"], *popt)

plt.figure()
plt.plot(df_fitting["Year"], df_fitting["Nigeria"], label="main data")
plt.plot(df_fitting["Year"], df_fitting["pop_exp"], label="fit data")

plt.legend()
plt.title("First fit attempt")
plt.xlabel("year")
plt.ylabel("population")
plt.show()
print()

# Estimating values for scale factor and exponential factor for a better fit

popt = [0.5e8, 0.01]
df_fitting["pop_exp"] = exp_growth(df_fitting["Year"], *popt)

plt.figure()
plt.plot(df_fitting["Year"], df_fitting["Nigeria"], label="main data")
plt.plot(df_fitting["Year"], df_fitting["pop_exp"], label="fit data")

plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Improved fit attempt")
plt.show()

popt = [0.5e8, 0.024]
df_fitting["pop_exp"] = exp_growth(df_fitting["Year"], *popt)

plt.figure()
plt.plot(df_fitting["Year"], df_fitting["Nigeria"], label="main data")
plt.plot(df_fitting["Year"], df_fitting["pop_exp"], label="fit data")

plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.title("Improved value")
plt.show()

# extracting the sigmas from the diagonal of the covariance matrix
sigma = np.sqrt(np.diag(covar))
print(sigma)

low, up = err.err_ranges(df_fitting["Year"], exp_growth, popt, sigma)

plt.figure()
plt.title("Exponential function")
plt.plot(df_fitting["Year"], df_fitting["Nigeria"], label="main data")
plt.plot(df_fitting["Year"], df_fitting["pop_exp"], label="fit data")

plt.fill_between(df_fitting["Year"], low, up, alpha=0.7)
plt.legend()
plt.xlabel("year")
plt.ylabel("population")
plt.show()

#Forecasting population of Nigeria
print()
print("Forecasted population of Nigeria")
low, up = err.err_ranges(2030, exp_growth, popt, sigma)
print("2030 between ", low, "and", up)
low, up = err.err_ranges(2040, exp_growth, popt, sigma)
print("2040 between ", low, "and", up)
low, up = err.err_ranges(2050, exp_growth, popt, sigma)
print("2050 between ", low, "and", up)

# Error ranges of + or -
print()
print('Forecasted population of Nigeria + or - : \n')
low, up = err.err_ranges(2030, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2030:", mean, "+/-", pm)

low, up = err.err_ranges(2040, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2040:", mean, "+/-", pm)

low, up = err.err_ranges(2050, exp_growth, popt, sigma)
mean = (up+low) / 2.0
pm = (up-low) / 2.0
print("2050:", mean, "+/-", pm)