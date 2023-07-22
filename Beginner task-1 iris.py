#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Import Libraries/Packages
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
import warnings
warnings.filterwarnings('ignore')


# In[1]:


import pandas as pd

# Load the csv file into a DataFrame
dataset = pd.read_csv("Iris.csv")

dataset


# In[3]:


# Shape of Dataset
dataset.shape


# In[2]:


# Display the first few rows of the DataFrame
dataset.head()


# In[4]:


# Dataset Columns
dataset.columns


# In[5]:


#Checking Null Values
dataset.isnull().sum()


# In[6]:


#Dataset Summary
dataset.info()


# In[7]:


#Dataset Statistical Summary
dataset.describe()


# In[8]:


# To display no. of samples on each class.
dataset['Species'].value_counts()


# In[9]:


dataset['Species'].value_counts().plot(kind = 'pie',  autopct = '%1.1f%%', shadow = True, explode = [0.09,0.09,0.09])


# In[10]:


dataset.corr()


# In[11]:


import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(9, 7))
sns.heatmap(dataset.corr(), cmap='CMRmap', annot=True, linewidths=2)
plt.title("Correlation Graph", size=20)
plt.show()


# In[12]:


#Label encoding for categorical variables
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Species'] = le.fit_transform(dataset['Species'])
dataset.head()


# In[13]:


dataset['Species'].unique()


# In[15]:


import pandas as pd
from sklearn.model_selection import train_test_split

# Load the CSV file into a DataFrame
df = pd.read_csv("Iris.csv")

features = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X = df.loc[:, features].values
Y = df['Species'].values

# Split the dataset into training and test sets
X_Train, X_Test, Y_Train, Y_Test = train_test_split(X, Y, test_size=0.2, random_state=0)

print(X_Train.shape)


# In[16]:


Y_Train.shape


# In[17]:


X_Test.shape


# In[18]:


Y_Test.shape


# In[19]:


# Feature Scaling to bring all the variables in a single scale.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_Train = sc.fit_transform(X_Train)
X_Test = sc.transform(X_Test)


# Importing some metrics for evaluating  models.
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import  classification_report
from sklearn.metrics import confusion_matrix


# In[20]:


from sklearn.linear_model import LogisticRegression
log_model= LogisticRegression(random_state = 0)
log_model.fit(X_Train, Y_Train)

# model training
log_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_log_res=log_model.predict(X_Test)

     

print(Y_Pred_Test_log_res)
log_model = LogisticRegression(random_state=0, max_iter=100)


# In[21]:


from sklearn import metrics

accuracy = metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)
print("Accuracy:", accuracy * 100)


# In[22]:


from sklearn.metrics import  classification_report
print(classification_report(Y_Test, Y_Pred_Test_log_res))


# In[23]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_Test,Y_Pred_Test_log_res )


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors=3, weights='distance', algorithm='auto')

# Importing KNeighborsClassifier 
from sklearn.neighbors import KNeighborsClassifier
knn_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)

# model training
knn_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_knn=knn_model.predict(X_Test)
     

# model training
log_model.fit(X_Train, Y_Train)


# In[25]:


Y_Pred_Test_knn


# In[26]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_knn)*100)


# In[27]:


print(classification_report(Y_Test,Y_Pred_Test_knn))


# In[28]:


confusion_matrix(Y_Test, Y_Pred_Test_knn)


# In[29]:


from sklearn.tree import DecisionTreeClassifier
dec_tree = DecisionTreeClassifier(criterion='entropy',splitter='best',max_depth=6)

# model training
dec_tree.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_dtr=dec_tree.predict(X_Test)
     

Y_Pred_Test_dtr


# In[30]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_dtr)*100)

print(classification_report(Y_Test, Y_Pred_Test_dtr))
     


# In[31]:


confusion_matrix(Y_Test, Y_Pred_Test_dtr)


# In[32]:


from sklearn.naive_bayes import GaussianNB
nav_byes = GaussianNB()

# model training
nav_byes.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_nvb=nav_byes.predict(X_Test)
     

Y_Pred_Test_nvb


# In[33]:


print("Accuracy:",metrics.accuracy_score(Y_Test, Y_Pred_Test_nvb)*100)


print(classification_report(Y_Test, Y_Pred_Test_nvb))


# In[34]:


confusion_matrix(Y_Test,Y_Pred_Test_nvb )


# In[35]:


from sklearn.svm import SVC
svm_model=SVC(C=500, kernel='rbf')

# model training
svm_model.fit(X_Train, Y_Train)

# Predicting
Y_Pred_Test_svm=svm_model.predict(X_Test)
     

Y_Pred_Test_svm


# In[36]:


print("Accuracy:",metrics.accuracy_score(Y_Test,Y_Pred_Test_svm)*100)
     
print(classification_report(Y_Test, Y_Pred_Test_svm))


# In[37]:


confusion_matrix(Y_Test,Y_Pred_Test_svm )


# In[38]:


print("Accuracy of Logistic Regression Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_log_res)*100)
print("Accuracy of KNN Model:",metrics.accuracy_score(Y_Test,Y_Pred_Test_knn)*100)
print("Accuracy of Decision Tree Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_dtr)*100)
print("Accuracy of Naive Bayes Model:",metrics.accuracy_score(Y_Test, Y_Pred_Test_nvb)*100)
print("Accuracy of SVM Model:",metrics.accuracy_score(Y_Test,Y_Pred_Test_svm)*100)


# In[ ]:




