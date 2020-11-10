#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append(".")
from base_layer import SoftmaxWithLoss, MatMul
import numpy as np

class SimpleSkipGram:
    def __init__(self, vacab_size, hidden_size):
        V, H  = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01*np.random.randn(V,H).astype("f")
        W_out = 0.01*np.random.randn(H,V).astype("f")
        
        # layer 생성
        self.in_layer = MatMul(W_in)
        self.out_layer = MatMul(W_out)
        self.loss_layer1 = SoftmaxWithLoss()
        self.loss_layer2 = SoftmaxWithLoss()
        
        layers = [self.in_layer, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 인스턴스 변수에 단어의 분산 표현을 저장
        self.words_vecs = W_in
        
    def forward(self, contexts, target):
        h = self.in_layer.forward(target)
        s = self.out_layer.forward(h)
        l1 = self.loss_layer1.forward(s, contexts[:,0])
        l2 = self.loss_layer2.forward(s, contexts[:,1])
        loss = l1+l2
        return loss
        
    def backward(self, dout=1):
        dl1 = self.loss_layer1.backward(dout)
        dl2 = self.loss_layer2.backward(dout)
        ds = dl1+dl2
        dh = self.out_layer.backward(ds)
        self.in_layer.backward(dh)
        return None
        
        

