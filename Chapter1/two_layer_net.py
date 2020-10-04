#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("..")
import numpy as np
from function.base_layer import Affine, Sigmoid, SoftmaxWithLoss

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        I,H,O = input_size, hidden_size, output_size
        
        # 가중치와 편향 초기화
        W1 = 0.01 * np.random.randn(I,H) # 표준편차 0.01로 맞춤
        b1 = np.zeros(H)
        W2 = 0.01 * np.random.randn(H,O)
        b2 = np.zeros(O)
        
        # 계층생성
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)]
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 가중치와 기울기를 리스트에 모음
        self.params, self.grads = [], []
        for layer in self.layers:
            # activation layer에서는 더해지는 param과 grads가 없음
            self.params += layer.params
            self.grads += layer.grads
    
    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x # 예측값
    
    def forward(self,x,t):
        score = self.predict(x)
        loss = self.loss_layer.forward(t, score)
        return loss # 손실값
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers): # 역순으로 진행
            dout = layer.backward(dout)
        return dout

