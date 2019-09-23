
# coding: utf-8

# In[5]:

import pandas as pd
from factor_analyzer import FactorAnalyzer


# In[4]:

df=pd.read_csv("input.csv")


# # Factor Analysis:

# In[ ]:

get_ipython().system('pip install factor_analyzer')


# In[6]:

fa = FactorAnalyzer()


# In[19]:

#For eigen values
fa.analyze(df, rotation="varimax")
ev, v = fa.get_eigenvalues()
ev


# In[20]:

#based on the eigen value we can determine the number of factors. Thumb rule is need to consider the factors based eigen>1.
#So we have eigen value >1 till factor 17
fa.analyze(df, 17, rotation="varimax")


# In[21]:

#Factor loadings
fa.loadings


# In[22]:

#communalities
fa.get_communalities()


# In[23]:

#get_factor_variance
fa.get_factor_variance()


# In[26]:

Fac_score=fa.get_scores(df)


# In[27]:

print (Fac_score.head())


# # Cluster Analysis using factor scores

# In[28]:

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score


# In[82]:

n_clusters=[5,10,15,16,17,18,19,20,25,30]
new=[]
for m in n_clusters:
    
    clusterer = KMeans(n_clusters=m, random_state=10)
    cluster_labels = clusterer.fit_predict(Fac_score)

# The silhouette_score gives the average value for all the samples.
    # This gives a perspective into the density and separation of the formed
    # clusters
    silhouette_avg = silhouette_score(Fac_score, cluster_labels)
    print("For n_clusters =", m,"The average silhouette_score is :", silhouette_avg)
    new.append([m,silhouette_avg])

    # Compute the silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(Fac_score, cluster_labels)


# In[164]:

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
df = pd.DataFrame(new)
df.columns=["nclusters","silhouette_coeff"]
ax = df.plot.line(x='nclusters', y='silhouette_coeff', rot=0, figsize=(20 ,4), legend=True, fontsize=10)


# In[165]:

plt.show()


# In[39]:

# Number of clusters
kmeans = KMeans(n_clusters=20,random_state=10)


# In[43]:

# Fitting the input data
kmeans = kmeans.fit(Fac_score)
# Getting the cluster labels
labels = kmeans.predict(Fac_score)
# Centroid values
centroids = kmeans.cluster_centers_

silhouette_avg = silhouette_score(Fac_score, labels)
print("For 20 clusters The average silhouette_score is :", silhouette_avg)



# In[155]:

clu=pd.DataFrame(labels)
clu.columns=["Cluster_membership"]


# In[163]:

fig = plt.figure(figsize=(8,8))
clu.groupby('Cluster_membership').Cluster_membership.count().plot.pie(autopct='%1.1f%%' )
plt.show()


# In[45]:

#copying the cluster membership to input data
Clustered_data=Fac_score
Clustered_data['clusters'] = labels


# In[46]:

Clustered_data.head()


# # Discriminent analysis

# In[47]:

X=Clustered_data


# In[49]:

print (X.head())


# In[50]:

y=X.pop('clusters')


# In[60]:

print ('\033[0m'+"Independant variable info:\n",X.head())
print ("\n")
print ("Dependant variable info :\n",y.head())


# In[65]:

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
clf = LinearDiscriminantAnalysis()
clf.fit(X, y)


# In[71]:

y_pred_class = clf.predict(X)


# In[73]:

from sklearn import metrics
metrics.accuracy_score(y,y_pred_class)

