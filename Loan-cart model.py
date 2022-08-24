#!/usr/bin/env python
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,roc_auc_score,roc_curve


# In[23]:


ln=pd.read_csv(r"E:\data science\GREAT LAKES\back to studies\videaos\revision\datamining\week 2  cart\Loan+Delinquent+Dataset.csv")


# In[24]:


ln.head()


# In[25]:


ln.drop(["ID","delinquent"],axis=1,inplace=True)


# In[26]:


ln.head()


# In[27]:


ln.info()


# In[28]:


pd.crosstab(ln.Sdelinquent,ln.term,margins=True)


# In[29]:


g36=1-(np.square(3168/10589)+np.square(7421/10589))
g60=1-(np.square(659/959)+np.square(300/959))
print(g36)
print(g60)


# In[30]:


(g36*(10589/11548))+(g60*(959/11548))


# In[31]:


ln.Sdelinquent.value_counts()


# In[32]:


1-(np.square(7721/11548)+np.square(3827/11548))


# In[33]:


for i in ln:
    ln[i]= pd.Categorical(ln[i]).codes


# In[34]:


ln.head()


# In[35]:


x=ln.drop("Sdelinquent",axis=1)
y=ln["Sdelinquent"]

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)


# In[36]:


x_train.shape


# In[37]:


x_test.shape


# In[38]:


y_train.shape


# In[39]:


y_test.shape


# In[40]:


dc=DecisionTreeClassifier()
dc.fit(x_train,y_train)


# In[47]:


predict


# In[48]:


pd.crosstab(y_test,predict)


# In[54]:


confusion_matrix(y_test,predict)


# In[55]:


print(classification_report(y_test,predict))


# In[ ]:




