#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
import numpy as np
sys.path.append(os.path.join("..","master"))
from common.layers import Softmax

class AttentionWeight:
    # weight 를 구하는 class
    def __init__(self):
        # 파라미터가 없기 때문에 파라미터와 grads가 []로 설정됨
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
        
    def forward(self, hs, h):
        N, T, H = hs.shape
        
        hr = h.reshape(N,1,H).repeat(T,axis=1)
        # 내적
        t = hs*hr
        s = np.sum(t, axis=2)
        # 정규화
        a = np.softmax.forward(s)
        
        self.cache = (hs,hr)
        return a
    
    def backward(self, da):
        hs, hr = self.cache
        N,T,H = hs.shape
        
        ds = self.softmax.backward(da)
        dt = ds.reshape(N,T,1).repeat(H, axis=2) # sum의 역전파
        dhs = dt * hr
        dhr = dt * ds
        dh = np.sum(dhr, axis=1) # repeat의 역전파
        
        return dhs, dh

class WeightedSum:
    # weight와 encoder의 hs간 weighted sum을 구하는 class
    def __init__(self):
        # 학습하는 파라미터가 없기 때문에 파라미터와 gradient list가 []으로 설정됨
        self.params, self.grads = [], []
        self.cache = None
        
    def forward(self, hs, a):
        N, T, H = hs.shape
        
        # weighted sum
        ar = a.reshape(N,T,1).repeat(H, axis=2)
        t = hs * ar
        c = np.sum(t, axis=1)
        
        self.cache = (hs, ar)
        return c
    
    def backward(self, dc):
        hs, ar = self.cache
        N, T, H = hs.shape
        
        # T 방향으로 더해졌었기 때문에 T방향으로 repeat
        dt = dc.reshape(N,1,H).repeat(T, axis=1) # sum의 역전파
        dar = dt * hs # 가중치 matrix에 대한 미분값
        dhs = dt * ar # hs matrix에 대한 미분값
        
        # H 방향으로 repeat 되었었기 때문에 H방향으로 sum
        da = np.sum(dar, axis=2) # repeat의 역전파
        
        return dhs, da

class AttentionWeight:
    # weight 를 구하는 class
    def __init__(self):
        # 파라미터가 없기 때문에 파라미터와 grads가 []로 설정됨
        self.params, self.grads = [], []
        self.softmax = Softmax()
        self.cache = None
        
    def forward(self, hs, h):
        N, T, H = hs.shape
        
        hr = h.reshape(N,1,H).repeat(T,axis=1)
        # 내적
        t = hs*hr
        s = np.sum(t, axis=2)
        # 정규화
        a = self.softmax.forward(s)
        
        self.cache = (hs,hr)
        return a
    
    def backward(self, da):
        hs, hr = self.cache
        N,T,H = hs.shape
        
        ds = self.softmax.backward(da)
        dt = ds.reshape(N,T,1).repeat(H, axis=2) # sum의 역전파
        dhs = dt * hr
        dhr = dt * hs
        dh = np.sum(dhr, axis=1) # repeat의 역전파
        
        return dhs, dh
    
class Attention:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_weight_layer = AttentionWeight()
        self.weight_sum_layer = WeightedSum()
        self.attention_weight = None
        
    def forward(self, hs, h):
        a = self.attention_weight_layer.forward(hs, h)
        out = self.weight_sum_layer.forward(hs, a)
        self.attention_weight = a
        return out
    
    def backward(self, dout):
        dhs0, da = self.weight_sum_layer.backward(dout)
        dhs1, dh = self.attention_weight_layer.backward(da)
        dhs = dhs0+dhs1
        return dhs, dh
    
class TimeAttention:
    def __init__(self):
        self.params, self.grads = [], []
        self.layers = None
        self.attention_weights = None
        
    def forward(self, hs_enc, hs_dec):
        """
        hs_enc : encoder에서 출력된 hs matrix
        hs_dec : decoder에서 출력된 hs matrix
        """
        N, T, H = hs_dec.shape
        out = np.empty_like(hs_dec) # 최종적으로 나와야하는 matrix 크기
        self.layers = []
        self.attention_weights = []
        
        for t in range(T):
            layer = Attention()
            out[:,t,:] = layer.forward(hs_enc, hs_dec[:,t,:])
            self.layers.append(layer)
            self.attention_weights.append(layer.attention_weight)
            
        return out
    
    def backward(self, dout):
        N, T, H = dout.shape
        dhs_enc = 0
        dhs_dec = np.empty_like(dout)
        
        for t in range(T):
            layer = self.layers[t]
            dhs, dh = layer.backward(dout[:,t,:])
            dhs_enc += dhs # 모두 같은 dhs가 사용되기 때문에 더해줌
            dhs_dec[:,t,:] = dh # 사용되는 dh가 다르기 때문에 각각 배정해줌
            
        return dhs_enc, dhs_dec

