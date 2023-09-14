#!/usr/bin/env python
# coding: utf-8

# In[11]:


## Homework 2 Problem 4 Coding


# In[12]:


## Problem 4.a


# In[13]:


import math


# In[14]:


import numpy as np


# In[15]:


import matplotlib.pyplot as plt


# In[16]:


t = np.arange(0,np.pi+np.pi/100,np.pi/30)


# In[17]:


t


# In[18]:


y = np.array(np.cos(t))


# In[19]:


y


# In[20]:


t[0]


# In[21]:


y[0]


# In[22]:


n=30
sum=0
for i in range(0,n):
    sum=sum+(t[i]*y[i])
print("the sum is: S",sum)


# In[23]:


## Problem 4.b


# In[26]:


theta = np.linspace(0,2*np.pi,num=100)
R = 1.2
deltar = 0.1
f = 15
p = 0


# In[27]:


x=R*(1+deltar*np.sin(f*theta+p))*np.cos(theta)
y=R*(1+deltar*np.sin(f*theta+p))*np.sin(theta)


# In[32]:


plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('4.b Figure 1')
plt.show()


# In[36]:


for i in range(0,11):
    R2=i
    deltar2=0.05
    f2=(2+i)
    p2=np.random.uniform(0,2)
    x2=R2*(1+deltar2*np.sin(f2*theta+p2))*np.cos(theta)
    y2=R2*(1+deltar2*np.sin(f2*theta+p2))*np.sin(theta)
    plt.plot(x2,y2)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('4.b Figure 2')
plt.show()


# In[ ]:




