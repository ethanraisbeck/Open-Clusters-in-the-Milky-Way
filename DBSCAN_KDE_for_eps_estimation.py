#%% Create dataframe
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import NearestNeighbors

'''Load data into dataframe'''

gaia_request = np.load('gaia_request.npy', allow_pickle=True)

data = pd.DataFrame(gaia_request,
                    columns=['ra','dec','parallax','pmra','pmdec'])

'''pre-processing and normalisation'''

imputer = SimpleImputer(strategy='mean')
data_ = pd.DataFrame(imputer.fit_transform(data),
                       columns=['ra','dec','parallax','pmra','pmdec'])

#%% Find KNN distances
'''Sample a fraction of the data'''

frac = (int(len(data)*1))
sample = data_.sample(frac, random_state=0)

'''Find KNN distances'''

neighb = NearestNeighbors(n_neighbors=9) 
nbrs=neighb.fit(data_) 
distances,indices=nbrs.kneighbors(sample) 

'''Plot hist'''
bins = np.arange(0,1.2,0.01)
plt.figure(figsize=(10,6))
plt.title('9^{th} Nearest Neighbour Distance')
plt.ylabel('Data points')
plt.xlabel('Epsilon')
plt.hist(distances[:,8], bins=bins)
plt.show()


#%%  Gaussian Kernel Density Bandwidth CV 
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import GridSearchCV
import time


def get_KDE_bandwidth(start,finish,step):

    startTime = time.time()
    
    grid = GridSearchCV(KernelDensity(kernel='gaussian', algorithm='kd_tree'),
                        {"bandwidth": np.arange(start,finish,step)}, 
                        n_jobs=-1, cv=5, )
    grid.fit(sample)
    
    print("best bandwidth: {0}".format(grid.best_estimator_.bandwidth))
    
    executionTime = (time.time() - startTime)
    print('Execution time in seconds: ' + str(executionTime))
    
    return grid.best_estimator_


kde = get_KDE_bandwidth(1,1.32,0.06)


#%%  Gaussian Kernel Density Estimation and Resampling

'''Find KNN distances'''

def get_epsilon(kde):
    
    dfs = []
    for i in range(30):
        resample = kde.sample(len(sample))
        neighb = NearestNeighbors(n_neighbors=9) 
        nbrs = neighb.fit(resample) 
        distances_res , indices = nbrs.kneighbors(resample) 
        dfs.append(pd.DataFrame(sorted(distances_res[:,8])))
    
    
    '''Average resampled histograms'''
    sum_ = 0
    for i in range(len(dfs)):
        sum_ += dfs[i]
    
    resample_average = sum_/len(dfs)
    
    '''Calculate final epsilon'''
    e_rand = np.average([min(i) for i in np.array(dfs)])
    
    e_knn = min(distances[:,8])
    
    e_avg = (e_rand + e_knn)/2
    
    return e_avg, resample_average

#%%  Plot histogram with resampled data

'''Plot histogram'''
bins = np.arange(0,1.5,0.01)

plt.rc('xtick', labelsize=12) 
plt.rc('ytick', labelsize=12)
plt.figure(figsize=(10,6))

plt.ylabel('Number of stars', fontdict={'fontsize': 14})
plt.xlabel('$9^{th}$ Nearest Neighbour Distance', fontdict={'fontsize': 14})
plt.hist(distances[:,8], bins=bins, histtype='step', color='blue')
plt.hist(get_epsilon(kde)[1], bins=bins, histtype='step', color='red')
plt.axvline(get_epsilon(kde)[0], label = 'axvline - full height', color='green')
plt.legend(['GAIA data','Resampled data','Determined value of ε'],
           fontsize='medium', loc='upper right')
plt.show()

print('Calculated epsilon: ' + str('{0:.3f}'.format(get_epsilon(kde)[0])))






