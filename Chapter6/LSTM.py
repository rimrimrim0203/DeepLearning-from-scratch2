#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
import sys
sys.append(".")
from base_function import *
from base_layer import *

class LSTM:
    def __init__(self, Wx, Wh, b):
        # 4개의 가중치가 담겨 있는 초기화 인수
        self.params = [Wx, Wh, b] # RNN과 파라미터 개수는 같지만 그 형상이 다름
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh),
                         np.zeros_like(b)]
        # 순전파 때 중간 결과를 보관했다가 역전파 계산에 사용하려는 용도의 인스턴스 변수
        self.cache = None
    
    def forward(self, x, h_prev, c_prev):
        """
        x : 현시각의 입력
        h_prev : 이전 시각의 은닉상태
        c_prev : 이전 시각의 memory cell
        """
        Wx, Wh, b = self.params
        N, H = h_prev.shape     # 이전 시각의 은닉상태의 크기를 기억
        
        A = np.matmul(x, Wx) + np.matmul(h_prev, Wh) + b # 4가지를 행렬곱으로 한번에 계산
        
        # slice
        f = A[:,:H]
        g = A[:,H:2*H]
        i = A[:,2*H:3*H]
        o = A[:,3*H:]
        
        f = sigmoid(f)
        g = tanh(g)
        i = sigmoid(i)
        o = sigmoid(o)
        
        # 원소곱을 통한 계산
        c_next = f*c_prev + g*i
        h_next = tanh(c_next)*o
        
        self.cache = (x, h_prev, c_prev, i, f, g, o, c_next)
        
        return h_next, c_next
    
    def backward(self, dh_next, dc_nex t):
        Wx, Wh, b = self.params # 파라미터 정리
        x, h_prev, c_prev, i, f, g, o, c_next = self.cache
        
        # C(t)에 대한 미분
        tanh_c_next = np.tanh(c_next)
        ds = dc_next + (dh_next*o) * (1-tanh_c_next ** 2) #다른 곳에서의 미분+이후에서의 미분*(tanh미분)
        
        # C(t-1)에 대한 미분
        dc_prev = ds*f
        
        di = ds*g
        df = ds*c_prev
        do = dh_next * tanh_c_next
        dg = ds*i
        
        # slide된 것을 합치기(열방향으로)
        dA = np.hstack((df, dg, di, do))
        
        dWh = np.dot(h_prev.T, dA)
        dWx = np.dot(x.T, dA)
        db  = dA.sum(axis=0)
        
        # gradient update
        self.grads[0][...] = dWx
        self.grads[1][...] = dWh
        self.grads[2][...] = db
        
        dx = np.dot(dA, Wx.T) 
        dh_prev = np.dot(dA, Wh.T)
        
        return dx, dh_prev, dc_prev

