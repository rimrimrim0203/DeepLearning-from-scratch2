#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import sys
sys.path.append("..")
from function.base_function import softmax, cross_entropy_error


# In[2]:


class Sigmoid:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None
    
    def forward(self, x):
        out = 1/(1+np.exp(-x))
        self.out = out
        return out
    def backward(self, dout):
        dx = dout*self.out*(1-self.out)
        return dx


# In[ ]:


class Affine:
    def __init__(self,W,b):
        self.params, self.grads = [W,b], [np.zeros_like(W),np.zeros_like(b)]
        self.x = None
        
    def forward(self,x):
        W,b = self.params
        out = np.matmul(x,W)+b
        self.x = x
        return out
    
    def backward(self, dout):
        W,b = self.params
        dW = np.matmul(self.x.T, dout)
        dx = np.matmul(dout,W.T)
        db = np.sum(dout, axis=0)
        
        self.grads[0][...] = dW
        self.grads[1][...] = db
        
        return dx
        


# In[5]:


class SoftmaxWithLoss:
    def __init__(self):
        self.params, self.grads = [],[]
        self.y = None
        self.t = None
        
    def forward(self,x,t):
        self.t = t
        self.y = softmax(x)
        
        # 정답 레이블이 원핫 벡터인 경우 정답의 인덱스로 변환
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)
        
        loss = cross_entropy_error(self.t, self,y)
        return loss
    
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        
        # gradient : softmax로 예측한 값에서 target 값을 뺀 값
        dx = self.y.copy()
        dx[np.arange(batch_size),self.t] -= 1
        dx *= dout
        
        return dx


# In[1]:


class MatMul:
    def __init__(self,W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        
    def forward(self,x):
        w, = self.params
        out = np.dot(x,w)
        self.x = x
        return out
    
    def backward(self, dout):
        w, = self.params
        dx = np.dot(dout,w.T)
        dw = np.dot(self.x.T, dout)
        self.grads[0][...] = dw
        return dx


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




