#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append(".")
from function.time_layers import *
import pickle

class Rnnlm:
    def __init__(self, vocab_size = 10000, wordvec_size = 100, hidden_size = 100):
        """
        vocab_size : input word 개수
        wordvec_size : embedding size
        hidden size : hidden state column size
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn # 더 편하게 사용하기 위해 rn 으로 바꿈
        
        # 가중치 초기화
        embed_W = (rn(V,D)/100).astype("f") # 표준편차 1/100으로 설정
        lstm_Wx = (rn(D, 4*H) / np.sqrt(D)).astype("f") # xavier 초기화
        lstm_Wh = (rn(H, 4*H) / np.sqrt(H)).astype("f") # xavier 초기화
        lstm_b = np.zeros(4*H).astype("f")
        affine_W = (rn(H,V) / np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")
        
        # 계층 생성
        self.layers = [
            TimeEmbedding(embed_W),
            TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True),
            TimeAffine(affine_W, affine_b)
        ]
        self.loss_layer = TimeSoftmaxWithLoss()
        self.lstm_layer = self.layers[1] # lstm layer을 따로 저장(state 정의를 위해)
        
        # 모든 가중치와 기울기를 리스트에 모음
        self.params, self.grads = [], []
        for layer in self.layers:
            self.params += layer.params
            self.grads += layer.grads
            
    def predict(self, xs):
        for layer in self.layers:
            xs = layer.forward(xs)
        return xs
    
    def forward(self, xs, ts):
        score = self.predict(xs)
        loss = self.loss_layer.forward(score, ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.loss_layer.backward(dout)
        for layer in reversed(self.layers):
            dout = layer.backward(dout)
        return dout
    
    def reset_state(self):
        self.lstm_layer.reset_state()
        
    def save_params(self, file_name = "Rnnlm.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.params, f)
            
    def load_params(self, file_name = "Rnnlm.pkl"):
        with open(file_name, "rb") as f:
            self.params = pickle.load(f)

