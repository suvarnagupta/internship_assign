
# coding: utf-8

# # installing libraries
# import numpy as np
# import seaborn as sns
# import pandas as pd

# In[2]:


import numpy as np
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# In[10]:


datas = pd.read_csv(r"C:\Users\dell\Downloads\news\news.csv")


# In[11]:


datas.shape


# In[12]:


datas.head()


# In[13]:


lbl=datas.label


# In[7]:


lbl.head()


# In[16]:


x_train,x_test,y_train,y_test=train_test_split(datas['text'],lbl,test_size=0.2,random_state=7)


# In[18]:


#Dataflair - Innitiallise a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_datas=0.7)

#Dataflair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[19]:


#Dataflair - Innitiallise a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english',max_datas=0.7)

#Dataflair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[20]:


#Dataflair - Innitiallise a TfidfVectorizer
tfidf_vectorizer=TfidfVectorizer(stop_words='english')

#Dataflair - Fit and transform train set, transform test set
tfidf_train=tfidf_vectorizer.fit_transform(x_train)
tfidf_test=tfidf_vectorizer.transform(x_test)


# In[21]:


#DataFlair - Innitiallise a PassiveAggressiveClassifier
pc = PassiveAggressiveClassifier(max_iter=50)
pc.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pc.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {rounf(score*100,2)}%')


# In[22]:


#DataFlair - Innitiallise a PassiveAggressiveClassifier
pc = PassiveAggressiveClassifier(max_iter=50)
pc.fit(tfidf_train,y_train)

#DataFlair - Predict on the test set and calculate accuracy
y_pred=pc.predict(tfidf_test)
score=accuracy_score(y_test,y_pred)
print(f'Accuracy: {round(score*100,2)}%')


# In[23]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred,lbl=['FAKE','REAL'])


# In[24]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred, lbl=['FAKE','REAL'])


# In[25]:


#DataFlair - Build confusion matrix
confusion_matrix(y_test,y_pred)

