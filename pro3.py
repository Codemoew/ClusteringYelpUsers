import copy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = pd.read_csv("/Users/yang/Desktop/yelp.csv")

features_to_remove = ['user_id', 'name','yelping_since','average_stars']
for f in features_to_remove:
    del data[f]

data['likes'] = 0
data['compliments'] = 0
   
features_likes = ['useful', 'funny','cool']
for f in features_likes:
    data['likes'] += data[f]
    del data[f]

features_compliments = ['compliment_hot','compliment_more', 'compliment_profile', 
                        'compliment_cute','compliment_list', 'compliment_note', 
                        'compliment_plain','compliment_cool', 'compliment_funny', 
                        'compliment_writer','compliment_photos']
for f in features_compliments:
    data['compliments'] += data[f]
    del data[f]

for i,item in enumerate(data['elite'].values):
    if item=='None':
        data['elite'].values[i] = 0
    else:
        data['elite'].values[i] = item.count(',')+1

Data = data.values

from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
Data = scaler.fit_transform(Data)

#%%
def k_means(Data,k,b):
    user,feature = Data.shape
    #k = number of centeroids
    #initialize centers with points picked randomly in the dataset
    cen = Data[np.random.choice(user, k, replace=False)]
    cen = cen.astype(float)
    
    '''
    Dist_all = np.zeros([k,user])
    for c in range(k):
        Dist_all[c] = np.linalg.norm(Data - cen[c],axis=1)
    Dist_cen = np.amin(Dist_all,axis=0)
    
    Darg = sum(Dist_cen)/len(Dist_cen)
    print(Darg)
    '''  
    #mini-batch k-means
    count = np.zeros(k)

    diff = 1 
    i = 0 

    while diff>pow(10,-4):
        i+=1
        prev_cen = copy.deepcopy(cen)
        #take a subset of data x=b points
        rand_index = np.random.choice(user, b, replace=False)
    
        for ind in rand_index:
            #calculate the distances between the point and the centers
            dist = np.linalg.norm(cen - Data[ind],axis=1)
            #assign the point to its nearest center
            cen_ind = np.argmin(dist)
            count[cen_ind] += 1
            #update the center
            cen[cen_ind] = cen[cen_ind] + 1/count[cen_ind] * (Data[ind] - cen[cen_ind])
        diff = sum(np.linalg.norm(cen - prev_cen,axis=1))/k
        if i==1000:
            diff = 0

    Dist_all = np.zeros([k,user])
    for c in range(k):
        Dist_all[c] = np.linalg.norm(Data - cen[c],axis=1)
    Dist_cen = np.amin(Dist_all,axis=0)
    
    Darg = sum(Dist_cen)/len(Dist_cen)    
    Dmax = max(Dist_cen)
    Dmin = min(Dist_cen)
            
    print('k-means converges after {} iterations when k={}, with average distance={}'.format(i,k,Darg))
    
    return Darg,Dmax,Dmin

#%%
def k_means_plus(Data,k,b):
    user,feature = Data.shape
    #pick the first center uniformly at random
    cen = Data[np.random.choice(user, 1)]
    #compute the probabilities for all data points
    d = np.sum(np.square(Data - cen),axis=1)
    prob = d/sum(d)
    for i in range(1,k):
        new_cen = Data[np.random.choice(user,p=prob)]
        new_d = np.sum(np.square(Data - new_cen),axis=1)
        cen = np.vstack([cen,new_cen])
        d = np.amin(np.vstack([d,new_d]),axis=0)
        prob = d/sum(d)

    '''       
    Dist_all = np.zeros([k,user])
    for c in range(k):
        Dist_all[c] = np.linalg.norm(Data - cen[c],axis=1)
    Dist_cen = np.amin(Dist_all,axis=0)
    
    Darg = sum(Dist_cen)/len(Dist_cen)
    print(Darg)
    '''
    diff = 1 
    i = 0     
    count = np.zeros(k)
    while diff>pow(10,-4):
        i+=1
        prev_cen = copy.deepcopy(cen)
        #take a subset of data x=b points
        rand_index = np.random.choice(user, b, replace=False)
    
        for ind in rand_index:
            #calculate the distances between the point and the centers
            dist = np.linalg.norm(cen - Data[ind],axis=1)
            #assign the point to its nearest center
            cen_ind = np.argmin(dist)
            count[cen_ind] += 1
            #update the center
            cen[cen_ind] = cen[cen_ind] + 1/count[cen_ind] * (Data[ind] - cen[cen_ind])
        diff = sum(np.linalg.norm(cen - prev_cen,axis=1))/k
        if i==1000:
            diff = 0
            
    Dist_all = np.zeros([k,user])
    for c in range(k):
        Dist_all[c] = np.linalg.norm(Data - cen[c],axis=1)
    Dist_cen = np.amin(Dist_all,axis=0)
    
    Darg = sum(Dist_cen)/len(Dist_cen)    
    Dmax = max(Dist_cen)
    Dmin = min(Dist_cen)
    
    print('k-means++ converges after {} iterations when k={}, with average distance={}'.format(i,k,Darg))

    return Darg,Dmax,Dmin    

#k_means_plus(Data,5,100)


#%%
def my_k_means(Data,k,b):
    user,feature = Data.shape
    #find the most central point
    central = np.sum(Data,axis=0)/user
    #compute the distances of datapoints to the central point
    d = np.linalg.norm(Data - central,axis=1)
    #compute the PDF of the distances
    hist,edges = np.histogram(d,bins=10000)
    #choose k centeroids according to the PDF of distances
    dist = np.random.choice(edges[:10000],k,replace=False,p=hist/sum(hist))
    #print(dist)
    cen = np.zeros([k,feature])
    for i,item in enumerate(dist):
        ind = np.argmin(np.abs(d-item))
        cen[i] = Data[ind]

    #mini-batch k-means
    count = np.zeros(k)

    diff = 1 
    i = 0 

    while diff>pow(10,-4):
        i+=1
        prev_cen = copy.deepcopy(cen)
        #take a subset of data x=b points
        rand_index = np.random.choice(user, b, replace=False)
    
        for ind in rand_index:
            #calculate the distances between the point and the centers
            dist = np.linalg.norm(cen - Data[ind],axis=1)
            #assign the point to its nearest center
            cen_ind = np.argmin(dist)
            count[cen_ind] += 1
            #update the center
            cen[cen_ind] = cen[cen_ind] + 1/count[cen_ind] * (Data[ind] - cen[cen_ind])
        diff = sum(np.linalg.norm(cen - prev_cen,axis=1))/k
        if i==1000:
            diff = 0

    Dist_all = np.zeros([k,user])
    for c in range(k):
        Dist_all[c] = np.linalg.norm(Data - cen[c],axis=1)
    Dist_cen = np.amin(Dist_all,axis=0)
    
    Darg = sum(Dist_cen)/len(Dist_cen)    
    Dmax = max(Dist_cen)
    Dmin = min(Dist_cen)
            
    print('my_k_means converges after {} iterations when k={}, with average distance={}'.format(i,k,Darg))
    
    return Darg,Dmax,Dmin

#%%
choices_of_k = [5,50,100,250,500]
l = len(choices_of_k)
max0 = []
min0 = []
arg0 = []
max1 = []
min1 = []
arg1 = []
max2 = []
min2 = []
arg2 = []

for k0 in choices_of_k:
    a,b,c = k_means(Data,k0,100)
    a0,b0,c0 = k_means_plus(Data,k0,100)
    a1,b1,c1 = my_k_means(Data,k0,100)

    arg0.append(a)
    max0.append(b)
    min0.append(c)
    
    arg1.append(a0)
    max1.append(b0)
    min1.append(c0)
    
    arg2.append(a1)    
    max2.append(b1)    
    min2.append(c1)
  
plt.plot(choices_of_k,arg0,'bs',label='k-means')
plt.plot(choices_of_k,arg1,'ro',label='k-means++') 
plt.plot(choices_of_k,arg2,'gx',label='my_k_means')
plt.legend(loc='best')
plt.title('average distances VS number of centroids')
plt.xlabel('k')
plt.ylabel('average distances')
plt.show()
 
plt.plot(choices_of_k,max0,'bs',label='k-means')
plt.plot(choices_of_k,max1,'ro',label='k-means++') 
plt.plot(choices_of_k,max2,'gx',label='my_k_means')
plt.legend(loc='best')
plt.title('maximum distances VS number of centroids')
plt.xlabel('k')
plt.ylabel('maximum distances')
plt.show()

plt.plot(choices_of_k,min0,'bs',label='k-means')
plt.plot(choices_of_k,min1,'ro',label='k-means++') 
plt.plot(choices_of_k,min2,'gx',label='my_k_means')
plt.legend(loc='best')
plt.title('minimum distances VS number of centroids')
plt.xlabel('k')
plt.ylabel('minimum distances')
plt.show()

