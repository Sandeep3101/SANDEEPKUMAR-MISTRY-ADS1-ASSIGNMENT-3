
"""
Created on Thu Jan 12 17:09:30 2023

@author: Sandeep
"""

# Import All The Python In-built Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.cluster import KMeans
import sklearn.metrics as skmet
import scipy.optimize as opt

def norm(array):
    """ Returns array normalised to [0,1]. Array can be a numpy array 
    or a column of a dataframe"""

    min_val = np.min(array) 
    max_val = np.max(array)
    
    scaled = (array-min_val) / (max_val-min_val)  
    
    return scaled

def norm_df(df, first=0, last=None):
    """ 
    Returns all columns of the dataframe normalised to [0,1] with the 
    exception of the first (containing the names)
    Calls function norm to do the normalisation of one column, but
    doing all in one function is also fine.
    First, last: columns from first to last (including) are normalised.
    Defaulted to all. None is the empty entry. The default corresponds 
    """
    
    # iterate over all numerical columns
    for col in df.columns[first:last]: # excluding the first column
        df[col] = norm(df[col])
        
    return df


# Function to read file
def readFile(filename):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        filename : csv filename
    
        Returns
        -------
        dataframe : variable for storing csv file
'''
    
    dataframe = pd.read_csv(filename)
    dataframe = dataframe.fillna(0.0)
    #dataframe = dataframe.drop(['Country Code', 'Indicator Name', 'Indicator Code'], axis = 1)
    
    return dataframe
 

# read file by calling function 
population = readFile("D:/Data Science Semester A/ADS 1/ADS 3 Clustering and Fitting Assignment/API_SP.POP.GROW_DS2_en_csv_v2_4770493.csv")
print(population)

# remove columns by drop functions
population_d = population.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960'], axis=1)
print("population_d")

# Transposing dataframe
population_growth = pd.DataFrame.transpose(population_d)
print(population_growth)

# populating header with header information
header1 = population_growth.iloc[0].values.tolist()
population_growth.columns = header1
print(population_growth)

# remove first two rows from dataframe
population_s = population_growth.iloc[2:]
print(population_s)

# setting figure size and dpi for better visualiation 
plt.figure(figsize=(8,6),dpi=1024)

# plotting data in scatter plot
plt.scatter(population_s["Jamaica"],population_s["Japan"],marker="o")
plt.xlabel("Jamaica" ,fontsize="14")
plt.ylabel("Japan", fontsize="14")
plt.title("Population Growth (%)", fontsize="16")
plt.tight_layout()
plt.savefig("Population Growth (%).png")
plt.show()

# creating a dataframe for two columns to store original values
pop_copy = population_s[["Jamaica","Japan"]].copy()

# finding maximum and minmum value of dataframe
max_val = pop_copy.max()
min_val = pop_copy.min()
pop_copy = (pop_copy - min_val) / (max_val - min_val) # operation of min and max

print(pop_copy)

# set up clusterer and number of clusters
ncluster = 5
kmeans = cluster.KMeans(n_clusters=ncluster)

# fitting the data where the results are stored in kmeans object
kmeans.fit(pop_copy)
 # labels is number of associated clusters
labels = kmeans.labels_

# extracting estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)


''' perform Silhoutte score Algorithm for  finding the number of clusters '''

for ic in range(2, 7):
    # set up kmeans and fit
    kmeans = cluster.KMeans(n_clusters=ic)
    kmeans.fit(pop_copy)     

    # extract labels and calculate silhoutte score
    labels = kmeans.labels_
    print (ic, skmet.silhouette_score(pop_copy, labels))
    
# Plot for three clusters
kmeans = cluster.KMeans(n_clusters=5)
kmeans.fit(pop_copy)     

# extract labels and cluster centres
labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(dpi = 1024)
# plotting scatter plot for clustering
plt.scatter(pop_copy["Jamaica"], pop_copy["Japan"], c=labels, cmap="Accent")
# show cluster centres
for ic in range(5):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
    
plt.xlabel("Jamaica")
plt.ylabel("Japan")
plt.title("Population Growth (%)")
plt.savefig("Population Growth silhoutte(%).png")
plt.show()


''' Perform Elbow method Algorithm for finding the number of clusters '''

x = pop_copy['Jamaica']
y = pop_copy['Japan']

data = list(zip(x, y))
inertias = []

for i in range(1,11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data)
    inertias.append(kmeans.inertia_)

plt.figure(dpi =1024)
plt.plot(range(1,11), inertias, marker='o')
plt.title('Elbow method',fontsize="16")
plt.xlabel('Number of clusters',fontsize="14")
plt.ylabel('Inertia',fontsize="14")
plt.tight_layout()
plt.savefig("Elbow method.png")
plt.show()

kmeans = KMeans(n_clusters=3)
kmeans.fit(data)

labels = kmeans.labels_
cen = kmeans.cluster_centers_

plt.figure(dpi = 1024)
# plotting scatter plot for clustering
plt.scatter(x, y, c=kmeans.labels_)
for ic in range(3):
    xc, yc = cen[ic,:]
    plt.plot(xc, yc, "dk", markersize=10)
   
plt.xlabel("Jamaica")
plt.ylabel("Japan")
plt.title("Population Growth (%)")
plt.savefig("Annual Population Growth (%).png")  
plt.show()

''' curve fitting '''    

# read file by calling function 
GDP = readFile("D:/Data Science Semester A/ADS 1/ADS 3 Clustering and Fitting Assignment/API_NY.GDP.MKTP.KD.ZG_DS2_en_csv_v2_4770541.csv")
print(GDP)

# Transpose dataframe
gdp_growth = GDP.transpose()
print(gdp_growth)

# populating header with header information
header2 = gdp_growth.iloc[0].values.tolist()
gdp_growth.columns = header2
print("\nGDP Growth Header: \n", gdp_growth)

# select particular column
gdp_growth = gdp_growth["United States"]
print("\nGDP growth after selecting particular column: \n", gdp_growth)

# rename column
gdp_growth.columns = ["GDP"]
print("\nRenamed GDP Growth: \n", gdp_growth)

# extracting particular rows
gdp_growth = gdp_growth.iloc[5:]
gdp_growth = gdp_growth.iloc[:-1]
print("\nGDP after selecting particular rows: \n", gdp_growth)

# resetn index of dataframe
gdp_growth = gdp_growth.reset_index()
print("\nGDP Growth reset index: \n", gdp_growth)

# rename columns
gdp_growth = gdp_growth.rename(columns={"index": "Year", "United States": "GDP"} )
print("\nGDP Growth after renamed columns: \n", gdp_growth)
print(gdp_growth.columns)

# plot line graph
plt.figure(dpi=1024)
gdp_growth.plot("Year", "GDP", label="GDP")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.title("GDP Growth")
plt.savefig('GDP Growth.png')
plt.show()

# curve fit with exponential function
def exponential(s, q0, h):
    '''
        Calculates exponential function with scale factor n0 and growth rate g.
    '''
    s = s - 1960.0
    x = q0 * np.exp(h*s)
    return x

# performing best fit in curve fit
print(type(gdp_growth["Year"].iloc[1]))
gdp_growth["Year"] = pd.to_numeric(gdp_growth["Year"])
print("\nGDP Growth Type: \n", type(gdp_growth["Year"].iloc[1]))
param, covar = opt.curve_fit(exponential, gdp_growth["Year"], gdp_growth["GDP"],
p0=(4.978423, 0.03))

#setting dpi for better visualisation
plt.figure(dpi=1024)

# plotting best fit
gdp_growth["fit"] = exponential(gdp_growth["Year"], *param)
gdp_growth.plot("Year", ["GDP", "fit"], label=["GDP (USA)", "fit"])
plt.xlabel("Year")
plt.ylabel("GDP")
plt.legend()
plt.title("GDP Growth")
plt.savefig('new GDP Growth.png')
plt.show()

# predict fit for nect 10 years
year = np.arange(1960, 2041)
print(year)
forecast_fit = exponential(year, *param)

#setting dpi for better visualisation
plt.figure(dpi=1024)

#plot a graph for prediction and setting label and title
plt.plot(gdp_growth["Year"], gdp_growth["GDP"], label="GDP (USA)")
plt.plot(year, forecast_fit, label="Forecast fit")
plt.xlabel("Year")
plt.ylabel("GDP")
plt.title("United States Growth Of GDP (%)")
plt.legend()
plt.savefig("GDP Growth (%).png")
plt.show()


# err_ranges function
def err_ranges(x, exponential, param, sigma):
    '''
        Calculates the upper and lower limits for the function, parameters and
        sigmas for single value or array x. Functions values are calculated for 
        all combinations of +/- sigma and the minimum and maximum is determined.
        Can be used for all number of parameters and sigmas >=1.
    
        This routine can be used in assignment programs.
    '''
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper
