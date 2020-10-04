#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[ ]:


def softmax(x):
    big = np.max(x, axis=1).reshape(-1,1)
    exp_x = np.exp(x-big)
    return exp_x/np.sum(exp_x, axis=1).reshape(-1,1)

def cross_entropy_error(t,y):
    epsilon = 1e-7
    if y.ndim == 1: # 2차원으로 바꿔줌
        y = y.reshape(1,y.size)
        t = t.reshape(1,t.size)
        
    batch_size = y.shape[0]
    
    # 정답 레이블이 원핫 벡터인 경우 정답의 인덱스로 변환
    if t.size == y.size:
        t = t.argmax(axis=1)
        
    return -np.sum(np.log(y[np.arange(batch_size),t] + 1e-7)) / batch_size

