#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append(os.path.join("..","master"))
sys.path.append(".")
from dataset import sequence
from common.base_model import *
from common.time_layers import *
import numpy as np

class Encoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V, D)/100).astype("f")
        lstm_Wx = (rn(D, 4*H)/np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4*H).astype("f")
        
        self.embed = TimeEmbedding(embed_W)
        # '긴 시계열 데이터'가 하나뿐인 문제를 다뤘을 때는 stateful = True
        # '짧은 시계열 데이터'가 여러개인 문제를 다룰 때는 stateful = False
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=False)
        
        self.params = self.embed.params + self.lstm.params
        self.grads = self.embed.grads + self.lstm.grads
        self.hs = None # 다음 계층으로 보내기 위한 hs
        
    def forward(self, xs):
        xs = self.embed.forward(xs)
        hs = self.lstm.forward(xs)
        self.hs = hs
        return hs[:, -1, :] # LSTM layer의 마지막 출력만 사용
    
    def backward(self, dh):
        # 끊기기 때문에 마지막 layer 빼고는 0의 gradient가 옴
        dhs = np.zeros_like(self.hs) 
        dhs[:, -1, :] = dh
        
        dout = self.lstm.backward(dhs)
        dout = self.embed.backward(dout)
        return dout

    


# In[ ]:


class Decoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        embed_W = (rn(V,D)/100).astype("f")
        lstm_Wx = (rn(D, 4*H)/np.sqrt(D)).astype("f")
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4*H).astype("f")
        affine_W = (rn(H,V)/np.sqrt(H)).astype("f")
        affine_b = np.zeros(V).astype("f")
        
        self.embed = TimeEmbedding(embed_W)
        
        # encoder의 상태를 갖도록 해야하기 때문에 stateful = True
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        
    def forward(self, xs, h):
        self.lstm.set_state(h) # encoder에서 출력된 hidden state를 설정
        
        out = self.embed.forward(xs)
        out = self.lstm.forward(out)
        score = self.affine.forward(out)
        return score
    
    def backward(self, dscore):
        dout = self.affine.backward(dscore)
        dout = self.lstm.backward(dout)
        dout = self.embed.backward(dout)
        dh = self.lstm.dh
        return dh # encoder에 적용하기 위해
    
    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)
        
        for _ in range(sample_size):
            x = np.array(sample_id).reshape(1,1) # 배치처리
            out = self.embed.forward(x)
            out = self.lstm.forward(out)
            score = self.affine.forward(out)
            
            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
            
        return sampled


# In[ ]:


class Seq2seq(BaseModel):
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = Decoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads
        
    def forward(self, xs, ts): # 훈련시 사용
        # 도착어가 decoder의 input과 output이 됨
        decoder_xs, decoder_ts = ts[:,:-1], ts[:,1:]
        h = self.encoder.forward(xs)
        # 출발어가 encoder의 input, h가 output
        score = self.decoder.forward(decoder_xs, h)
        loss = self.softmax.forward(score,decoder_ts)
        return loss
    
    def backward(self, dout=1):
        dout = self.softmax.backward(dout)
        dh = self.decoder.backward(dout)
        dout = self.encoder.backward(dh)
        return dout
    
    def generate(self, xs, start_id, sample_size): # 생성시 사용
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        return sampled

