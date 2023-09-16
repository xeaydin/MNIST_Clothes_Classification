#!/usr/bin/env python
# coding: utf-8

# # Image Classification & Dimensionality Reduction

# In[1]:


# Emir Aydın
# S020843


# In[2]:


import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report
from sklearn.decomposition import PCA
from sklearn.linear_model import Perceptron
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
np.random.seed(42) # DO NOT CHANGE
random.seed(42) # DO NOT CHANGE


# In[3]:


# Defining the function for loading MNIST dataset
def load_mnist(path, kind='train'):
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels


# In[4]:


# Initializing train and test datasets 
X_train, y_train = load_mnist('data', kind='train')
X_test, y_test = load_mnist('data', kind='t10k')


# In[5]:


X_train.shape


# In[6]:


#Normalization

X_train_std=X_train/255
X_test_std=X_test/255


# In[21]:


y_test.shape


# In[8]:


# defining a method for plotting x_train_std images
def image_plotting(image,n):
    plt.subplot(1,5,n)
    plt.axis("off")
    plt.imshow(image)


# In[9]:


# plotting the first five image of x_train_std
plt.figure(figsize=(50,20))
for i in range(5):
    w=np.resize(X_train_std[i,:],(28,28))
    image_plotting(w,i+1)
plt.show()


# # First Implementation of Algorithms and Predictions
# 

# In[10]:


# Initializing the K-Neighbors, Perceptron,
# SVM, Decision Tree and Random Forest classifiers
# with their hyper-parameters
knn = KNeighborsClassifier(n_neighbors=3, metric="minkowski", p=2)
perceptron = Perceptron(alpha=0.001)
svm = SVC()
dt = DecisionTreeClassifier()
rf = RandomForestClassifier()
# Declaring arrays to store classifiers and their names
models=np.array([knn,perceptron,svm,dt,rf])
names=np.array(["KNN","Perceptron","SVM","Decision Tree","Random Forest"])


# In[11]:


import seaborn as sns #importing seaborn for plotting heatmap
# Declaring a new method for plotting confusion matrixes by using heatmap from seaborn
def plot_confusion_matrix(y_test, y_predicted):  
    conf_mat = pd.DataFrame(confusion_matrix(y_test, y_predicted))  
    fig = plt.figure(figsize=(25, 15))  
    sns.heatmap(conf_mat, annot=True, annot_kws={"size": 16}, fmt="g")  
    plt.title("Confusion Matrix")  
    plt.xlabel("Predicted Label")  
    plt.ylabel("True Label")  
    plt.show() 


# In[12]:


predict_acc=np.array([]) # Array for storing accuracies of predictions
# Function for all 5 algorithms to make prediction with data that returns the accuracies
def predict_model(model,name,X1,X2,y1,y2):
    model.fit(X1,y1)
    y_predict=model.predict(X2)
    accur =accuracy_score(y_predict,y2)
    confusion_mat = confusion_matrix(y2,y_predict)
    plt.figure(figsize=(10,10))
    print("First 25 Instance Prediction of ",name)
    # Visualizing first 25 element of test values with their predicted labels
    for i in range(25):
        plt.subplot(5,5,i+1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(np.resize(X2[i,:],(28,28)), cmap=plt.cm.binary)
        plt.xlabel(y_predict[i])
    plt.show()
    print("Accuracy of the ",name,": ",accur) # Printing accuracy of the model
    print("Confusion Matrix of the ",name," : ") 
    plot_confusion_matrix(y2,y_predict) #Plotting confusion matrix of the model
    return accur


# In[13]:


# making predictions for all model in models array
for i in range(models.size):
    acc0=predict_model(models[i],names[i],X_train_std,X_test_std,y_train,y_test)
    predict_acc=np.append(predict_acc,acc0)


# In[14]:


# Printing the accuracies of algorithms after first implementation
for n in range(names.size):
    print("Accuracy of the ",names[n],": ",predict_acc[n])


# # Principal Component Analysis (PCA)

# In[15]:


# Declaring four different pca with different ratio of explained variances
pca_arr=np.array([])
pca_25 = PCA(n_components=0.25)
pca_arr=np.append(pca_arr,pca_25)
pca_50 = PCA(n_components=0.50)
pca_arr=np.append(pca_arr,pca_50)
pca_75 = PCA(n_components=0.75)
pca_arr=np.append(pca_arr,pca_75)
pca_95 = PCA(n_components=0.95)
pca_arr=np.append(pca_arr,pca_95)
pca_arr.size


# In[16]:


# Inside a for loop, the number of components
# to explain the given ratios of variance
# the dataset will be found and displayed
variance_arr=np.array([25,50,75,95])
components=np.array([])
for j in range(pca_arr.size):
    # Fitting the PCA object to the data
    pca=pca_arr[j]
    pca.fit(X_train_std)
    # Get the number of components required to explain P% of the variance
    n_components = pca.n_components_ # number of components
    components= np.append(components,n_components)
    print("Number of components required to explain ",variance_arr[j],"% of the variance: {}".format(n_components))


# In[17]:


components=components.astype('int32') # Changing the data type as integers
print(components)


# ## Implementing Algorithms after Finding Components
# 

# In[18]:


acc_arr=np.array([])
cm_arr=np.array([])
# Building a function make prediction for each 5 model by using each component which were found
# total 20 predictions
# This function will return accuracies and confusion metrixes of given models for each component
def build_model_withPca(model,name,X1,X2,Y1,Y2,components):
    model_acc=np.array([])
    model_cm=np.array([])
    for i in range(components.size):
        myPCA=PCA(n_components=components[i]) # Declaring a PCA for giving number of components
        myPCA.fit(X1) # Fitting the PCA object to the data
        X1_pca=myPCA.transform(X1) # Transforming X_train_std for PCA
        X2_pca=myPCA.transform(X2) # Transforming X_test_std for PCA
        model.fit(X1_pca, Y1)
        ypred=model.predict(X2_pca)
        cm=confusion_matrix(Y2,ypred)
        accuracy = accuracy_score(ypred,Y2)
        # Calculating and printing accuracy after PCA 
        print("Accuracy of the ",name," with ",components[i]," number of components: ",accuracy)
        plot_confusion_matrix(y_test,ypred) # Plotting confusion matrixes
        model_acc=np.append(model_acc,accuracy)
        model_cm=np.append(model_cm,cm)
    return model_acc,cm_arr
    


# In[19]:


# Prediction each 5 model for each number of components that have been found
for m in range(models.size):
    accs,cms=build_model_withPca(models[m],names[m],X_train_std,X_test_std,y_train,y_test,components)
    acc_arr=np.append(acc_arr,accs)
    cm_arr=np.append(cm_arr,cms)


# ## Accuracy Table

# In[20]:


# Resizing array of accuracies to create a table
acc_arr=np.resize(acc_arr,(5,4)) 

# Initializing a Pandas Dataframe for creating a table
# to display accuracies of the classification algorithms
# for each components
accuracy_table=pd.DataFrame(acc_arr, columns=['25% of variance', '50% of variance',
                                              '75% of variance','95% of variance'], index=names)
accuracy_table







