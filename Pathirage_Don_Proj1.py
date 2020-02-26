#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pandas as pd
import tkinter as tk
from tkinter import filedialog


# In[18]:


irisDf = pd.read_excel('Proj1DataSet.xlsx')


# In[19]:


labels = irisDf['species'].unique()
features = list(irisDf.columns.values)

#Assign class labels
irisDf.loc[irisDf['species'] == 'setosa', 'class'] = 0
irisDf.loc[irisDf['species'] == 'versicolor', 'class'] = 1
irisDf.loc[irisDf['species'] == 'virginica', 'class'] = 2

newColumns = ['sepL', 'sepW', 'petL', 'petW', 'class', 'species']
irisDf=irisDf.reindex(columns=newColumns)

markers = [] 
colors = []

# Mark by colors
for index, row in irisDf.iterrows():
    if row['species'] == 'setosa':
        colors.append('crimson')
    elif row['species'] == 'versicolor':
        colors.append('royalblue')
    else:
        colors.append('darkorange')
irisDf.head()


# In[20]:


plt.subplots(2,2,figsize=(15,8))
red = mpatches.Patch(color='red', label='Setosa')
blue = mpatches.Patch(color='royalblue', label='Versicolor')
orange = mpatches.Patch(color='darkorange', label='Virginica')

plt.subplot(2,2,1)
plt.scatter(np.linspace(0,len(irisDf['sepL'].values),len(irisDf['sepL'].values)),             irisDf['sepL'].values,marker='o',c=colors)
plt.title('Sepal Length of Species')
plt.xlabel('Sample')
plt.ylabel('Sepal Length')
plt.legend(handles=[red,blue,orange])

plt.subplot(2,2,2)
plt.scatter(np.linspace(0,len(irisDf['sepL'].values),len(irisDf['sepW'].values)),             irisDf['sepW'].values,marker='o',c=colors)
plt.title('Sepal Width of Species')
plt.xlabel('Sample')
plt.ylabel('Sepal Width')
plt.legend(handles=[red,blue,orange])

plt.subplot(2,2,3)
plt.scatter(np.linspace(0,len(irisDf['sepL'].values),len(irisDf['petL'].values)),             irisDf['petL'].values,marker='o',c=colors)
plt.title('Petal Length of Species')
plt.xlabel('Sample')
plt.ylabel('Petal Length')
plt.legend(handles=[red,blue,orange])

plt.subplot(2,2,4)
plt.scatter(np.linspace(0,len(irisDf['sepL'].values),len(irisDf['petW'].values)),             irisDf['petW'].values,marker='o',c=colors)
plt.title('Petal Width of Species')
plt.xlabel('Sample')
plt.ylabel('Petal Width')
plt.legend(handles=[red,blue,orange])


plt.tight_layout()


# In[21]:


#1. The features in this dataset are distinct enough to solve the problem, 
#   I expect the petal length and width features to be the better features 
#   for classifying the three flower species.


# In[22]:


#Compute stats of features

mins = []
maxs = []
means = []
var = []
intraVar = []
interVar = []

setosa = irisDf.loc[irisDf['species'] == 'setosa']
versicolor = irisDf.loc[irisDf['species'] == 'versicolor']
virginica = irisDf.loc[irisDf['species'] == 'virginica']

priorSetosa = len(setosa)/len(irisDf)
priorVersicolor = len(versicolor)/len(irisDf)
priorVirginica = len(virginica)/len(irisDf)


for i in range(0,len(features) - 1):
    
    mins.append(np.min(irisDf[features[i]]))
    maxs.append(np.max(irisDf[features[i]]))
    means.append(np.mean(irisDf[features[i]]))
    var.append(np.var(irisDf[features[i]]))
    
    intraVar.append( (priorSetosa*(np.var(setosa[features[i]])))                     + (priorVersicolor*(np.var(versicolor[features[i]])))                     + (priorVirginica*(np.var(virginica[features[i]]))) )
    
    interVar.append( (priorSetosa*(np.mean(setosa[features[i]]) - means[i])**2)                    + (priorVersicolor*(np.mean(versicolor[features[i]]) - means[i])**2)                    + (priorVirginica*(np.mean(virginica[features[i]]) - means[i])**2))


# In[23]:


statDf = pd.DataFrame([mins,maxs,means,var,intraVar,interVar],                       columns = ['sepL', 'sepW', 'petL', 'petW'],                       index=['Min', 'Max', 'Mean' , 'Var' , 'IntraClassVar' , 'InterClassVar'])


# In[24]:

print('Statistics of Dataset: ')
print(statDf.round(3))
print()

# In[25]:


## Correlation Coefficients
matfig = plt.figure(figsize=(15,8))
plt.matshow(irisDf[:-1].corr(),cmap='seismic', fignum=matfig.number)
plt.xticks(range(len(irisDf.columns)-1), irisDf.columns[:-1])
plt.yticks(range(len(irisDf.columns)-1), irisDf.columns[:-1])
plt.colorbar()


# In[26]:


# Features vs Class Labels
plt.subplots(2,2,figsize=(15,8))


plt.subplot(2,2,1)
plt.scatter(setosa['sepL'], np.ones([len(setosa),1]), marker='x', c='crimson')
plt.scatter(versicolor['sepL'], 2*np.ones([len(setosa),1]), marker='x', c='royalblue')
plt.scatter(virginica['sepL'], 3*np.ones([len(setosa),1]), marker='x', c='darkorange')
plt.xticks(np.arange(0,9,1))
plt.title('Sepal Length vs Class')
plt.legend(['Setosa','Virsicolor','Virginica'])

plt.subplot(2,2,2)
plt.scatter(setosa['sepW'], np.ones([len(setosa),1]), marker='x', c='crimson')
plt.scatter(versicolor['sepW'], 2*np.ones([len(setosa),1]), marker='x', c='royalblue')
plt.scatter(virginica['sepW'], 3*np.ones([len(setosa),1]), marker='x', c='darkorange')
plt.xticks(np.arange(0,9,1))
plt.title('Sepal Width vs Class')
plt.legend(['Setosa','Versicolor','Virginica'])

plt.subplot(2,2,3)
plt.scatter(setosa['petL'], np.ones([len(setosa),1]), marker='x', c='crimson')
plt.scatter(versicolor['petL'], 2*np.ones([len(setosa),1]), marker='x', c='royalblue')
plt.scatter(virginica['petL'], 3*np.ones([len(setosa),1]), marker='x', c='darkorange')
plt.xticks(np.arange(0,9,1))
plt.title('Petal Length vs Class')
plt.legend(['Setosa','Versicolor','Virginica'])

plt.subplot(2,2,4)
plt.scatter(setosa['petW'], np.ones([len(setosa),1]), marker='x', c='crimson')
plt.scatter(versicolor['petW'], 2*np.ones([len(setosa),1]), marker='x', c='royalblue')
plt.scatter(virginica['petW'], 3*np.ones([len(setosa),1]), marker='x', c='darkorange')
plt.xticks(np.arange(0,9,1))
plt.title('Petal Width vs Class')
plt.legend(['Setosa','Versicolor','Virginica'])


# In[27]:


def designMatrix(c1,c2,f):
    
    X = np.array(c1[f])
    t = np.ones([len(X),1])
    
    X = np.append(X,np.array(c2[f]),axis=0)
    t = np.append(t,-1*np.ones([len(np.array(c2[f])),1]),axis=0)
    X = np.hstack((X,np.ones(len(X)).reshape(len(X),1)))
    
    return X, t  

def predict(X,t,w):
    
    p = X@w
    
    p[p <= 0] = -1
    p[p > 0 ] = 1
    
    missclassed = np.where(p != t)[0]
    
    return missclassed

def closedFormLS(X,t):
    
    w = np.linalg.pinv(X) @ t
    missclassed = len(predict(X,t,w))
    
    return w, missclassed

def batchPerceptron(X,t,rho,epochs,e):
    
    epoch = 0
    w = np.zeros([X.shape[1],1])
    w2 = np.random.randn(X.shape[1],1)
    
    
    while(epoch < epochs and (np.absolute(w2-w)>e).any()):
        
        w2 = w
        missclassed = predict(X,t,w)
        
        for j in missclassed:
            w = w + rho*t[j]*X[j,:].reshape(len(w),1)
        epoch += 1
    
    if(epoch < epochs-1):
        print('Batch Perceptron converged in ' + str(epoch) + ' epochs')
    else:
        print('Batch Perceptron did not converge after ' + str(epochs) + ' epochs')
        
    return w, len(missclassed)   

def boundaries(perceptW,lsW):
    
    x = np.linspace(0,10,150).reshape(150,1)
    y1 = -(x*perceptW[0] + perceptW[2]) / perceptW[1]
    y2 = -(x*lsW[0] + lsW[2]) / lsW[1]
    
    return x, y1, y2

def lsBoundary(lsW):
    
    x = np.linspace(0,10,150).reshape(150,1)
    y = -(x*lsW[0] + lsW[2]) / lsW[1]
    
    return y


# In[28]:


# 1. Setosa vs Versi+Virgi for All Features
X1, t1 = designMatrix(c1=setosa,c2=pd.concat([versicolor,virginica]),f=features[:-2])

print("----------Setosa vs Versi+Virgi for All Features----------")
perceptW1, perceptMisclassed1 = batchPerceptron(X1,t1,rho=1e-3,epochs=250,e=10e-9)
lsW1, lsMisclassed1 = closedFormLS(X1,t1)

print('Batch Perceptron Weights: ' + str(perceptW1))
print('Number of Batch Perceptron Misclassifications: ' + str(perceptMisclassed1))
print()

print('Least-Squares Weights: ' + str(lsW1))
print('Number of Least-Squares Misclassifications: ' + str(lsMisclassed1))


# In[29]:


# 2. Setosa vs Versi+Virgi for Features 3 and 4

X2, t2 = designMatrix(c1=setosa,c2=pd.concat([versicolor,virginica]),f=[features[2],features[3]])

print("----------Setosa vs Versi+Virgi for Features 3 and 4----------")
perceptW2, perceptMisclassed2 = batchPerceptron(X2,t2,rho=1e-3,epochs=250,e=10e-9)
lsW2, lsMisclassed2 = closedFormLS(X2,t2)


print('Batch Perceptron Weights: ' + str(perceptW2))
print('Number of Batch Perceptron Misclassifications: ' + str(perceptMisclassed2))
print()

print('Least-Squares Weights: ' + str(lsW2))
print('Number of Least-Squares Misclassifications: ' + str(lsMisclassed2))

x, y1, y2 = boundaries(perceptW2,lsW2)

plt.figure(figsize=(15,8))
plt.scatter(setosa['petL'], setosa['petW'], marker='o', c='crimson')
plt.scatter(versicolor['petL'], versicolor['petW'], marker='o', c='royalblue')
plt.scatter(virginica['petL'], virginica['petW'], marker='o', c='darkorange')
plt.plot(x,y1)
plt.plot(x,y2)

plt.ylim([-1,3])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Setosa vs Versi+Virgi for Features 3 and 4')
plt.legend([ 'Perceptron', 'Least-Squares', 'Setosa','Versicolor','Virginica'])


# In[30]:


# 3. Virgi vs Versi+Setosa for All Features

X3, t3 = designMatrix(c1=virginica,c2=pd.concat([versicolor,setosa]),f=features[:-2])

print("----------Virgi vs Versi+Setosa for All Features----------")
perceptW3, perceptMisclassed3 = batchPerceptron(X3,t3,rho=1e-1,epochs=2000,e=10e-9)
lsW3, lsMisclassed3 = closedFormLS(X3,t3)


print('Batch Perceptron Weights: ' + str(perceptW3))
print('Number of Batch Perceptron Misclassifications: ' + str(perceptMisclassed3))
print()

print('Least-Squares Weights: ' + str(lsW3))
print('Number of Least-Squares Misclassifications: ' + str(lsMisclassed3))


# In[31]:


# 4. Virgi vs Versi+Setosa for Features 3 and 4

X4, t4 = designMatrix(c1=virginica,c2=pd.concat([versicolor,setosa]),f=[features[2],features[3]])

print("----------Virgi vs Versi+Setosa for Features 3 and 4----------")
perceptW4, perceptMisclassed4 = batchPerceptron(X4,t4,rho=1e-3,epochs=1000,e=10e-9)
lsW4, lsMisclassed4 = closedFormLS(X4,t4)

print('Batch Perceptron Weights: ' + str(perceptW4))
print('Number of Batch Perceptron Misclassifications: ' + str(perceptMisclassed4))
print()

print('Least-Squares Weights: ' + str(lsW4))
print('Number of Least-Squares Misclassifications: ' + str(lsMisclassed4))

x, y1, y2 = boundaries(perceptW4,lsW4)

plt.figure(figsize=(15,8))
plt.scatter(setosa['petL'], setosa['petW'], marker='o', c='crimson')
plt.scatter(versicolor['petL'], versicolor['petW'], marker='o', c='royalblue')
plt.scatter(virginica['petL'], virginica['petW'], marker='o', c='darkorange')

plt.plot(x,y1)
plt.plot(x,y2)

plt.ylim([-1,3])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Virgi vs Versi+Setosa for Features 3 and 4')
plt.legend([ 'Perceptron', 'Least-Squares', 'Setosa','Versicolor','Virginica'])
plt.rcParams["figure.figsize"] = [10,10]


# In[32]:


# 5. Setosa vs Versi vs Virgi for Features 3 and 4 with Multi-Class Least-Squares

f = [features[2],features[3]]
X5_1 = np.array(setosa[f])
t5_1 = np.repeat(np.array([1, 0, 0]).reshape(1,3),50,axis=0)

X5_2 = np.array(versicolor[f])
t5_2 = np.repeat(np.array([0, 1, 0]).reshape(1,3),50,axis=0)

X5_3 = np.array(virginica[f])
t5_3 = np.repeat(np.array([0, 0, 1]).reshape(1,3),50,axis=0)

X5 = np.concatenate((X5_1,X5_2,X5_3),axis=0)
X5 = np.hstack((X5,np.ones(len(X5)).reshape(len(X5),1)))

t5 = np.concatenate((t5_1,t5_2,t5_3),axis=0)

print("----------Setosa vs Versi vs Virgi for Features 3 and 4 with Multi-Class Least-Squares----------")

lsW5, junk = closedFormLS(X5,t5)
print('Multi-Class Least-Squares Weights:')
print(str(lsW5))
# Calculate misclassifications
predictions = []
truth = irisDf['class'].values.reshape(150,1).astype(int)
for i in range(len(t5)):
    
    p = X5[i]@lsW5
    predictions.append(np.argmax((np.abs(p))))
    
predictions = np.array(predictions).reshape(150,1)

print('Number of Least-Squares Misclassifications: ', np.sum(predictions!=truth))
print()

w1 = lsW5[:,0] - lsW5[:,1]
w2 = lsW5[:,0] - lsW5[:,2]
w3 = lsW5[:,2] - lsW5[:,1]

x = np.linspace(0,10,150).reshape(150,1)
y1 = lsBoundary(w1)
y2 = lsBoundary(w2)
y3 = lsBoundary(w3)

plt.figure(figsize=(15,8))
plt.scatter(setosa['petL'], setosa['petW'], marker='o', c='crimson')
plt.scatter(versicolor['petL'], versicolor['petW'], marker='o', c='royalblue')
plt.scatter(virginica['petL'], virginica['petW'], marker='o', c='darkorange')

plt.plot(x,y1)
plt.plot(x,y2)
plt.plot(x,y3)

plt.ylim([-1,3])
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.title('Setosa vs Versi vs Virgi for Features 3 and 4 with Multi-Class Least-Squares')
plt.legend(['D1','D2','D3','Setosa','Versicolor','Virginica'])
plt.rcParams["figure.figsize"] = [10,10]

plt.show()


# In[ ]:




