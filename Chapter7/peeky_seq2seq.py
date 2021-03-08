#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
import os
sys.path.append(os.path.join("..","master"))
sys.path.append(".")
from seq2seq import Seq2seq, Encoder
import numpy as np
from common.time_layers import *
    
class PeekyDecoder:
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        """
        vocab_size : 총 단어 개수(중복X)
        wordvec_size : embedding size
        hidden_size : hidden size
        """
        V, D, H = vocab_size, wordvec_size, hidden_size
        rn = np.random.randn
        
        # input에 encoder에서 출력되는 h가 추가되기 때문에 사이즈가 커짐
        # 원래 embedding layer에서 출력되는 데이터 크기 : (batch_size, D)
        # 변경 후 embedding layer에서 출력되는 데이터 크기 : (batch_size , D+H)
        embed_W = (rn(V,D)/100).astype("f")
        lstm_Wx = (rn(H+D, 4*H)/np.sqrt(H+D)).astype("f")
        lstm_Wh = (rn(H, 4*H)/np.sqrt(H)).astype("f")
        lstm_b = np.zeros(4*H).astype("f")
        affine_W = (rn(H+H, V)/np.sqrt(H+H)).astype("f")
        affine_b = np.zeros(V).astype("f")
        
        self.embed = TimeEmbedding(embed_W)
        self.lstm = TimeLSTM(lstm_Wx, lstm_Wh, lstm_b, stateful = True)
        self.affine = TimeAffine(affine_W, affine_b)
        
        self.params, self.grads = [], []
        for layer in (self.embed, self.lstm, self.affine):
            self.params += layer.params
            self.grads += layer.grads
        self.cache = None
        
    def forward(self,xs,h):
        N, T = xs.shape
        N, H = h.shape
        
        # encoder에서 출력된 vector h로 set함
        self.lstm.set_state(h)
        
        out = self.embed.forward(xs)
        hs = np.repeat(h, T, axis=0).reshape(N,T,H)
        # LSTM layer의 input을 변경시켜줌
        out = np.concatenate((hs, out), axis=2)
        
        out = self.lstm.forward(out)
        # Affine layer의 input을 변경시켜줌
        out = np.concatenate((hs, out), axis=2)
        
        score = self.affine.forward(out)
        self.cache = H
        return score
    
    def backward(self, dscore):
        H = self.cache
        
        dout = self.affine.backward(dscore)
        # concat 시켜주기 때문에 concat 시킨 input data의 gradient는 앞 계층으로 전달해주지 않음
        dout, dhs0 = dout[:,:,H:], dout[:,:,:H]
        dout = self.lstm.backward(dout)
        dembed, dhs1 = dout[:,:,H:], dout[:,:,:H]
        self.embed.backward(dembed)
        
        # dh 취합
        dhs = dhs0 + dhs1
        dh = self.lstm.dh + np.sum(dhs, axis=1) # 같은 layer의 시각별 gradient를 모두 합친후 첫번째 시각의 ds랑 합쳐줌
        return dh
        
    def generate(self, h, start_id, sample_size):
        sampled = []
        char_id = start_id
        self.lstm.set_state(h)
        
        H = h.shape[1]
        # 한개의 데이터만, 첫번째 시각 데이터만 넣어서 생성해주기 때문에 h의 shape 변환
        peeky_h = h.reshape(1,1,H)
        for _ in range(sample_size):
            x = np.array([char_id]).reshape((1,1))
            out = self.embed.forward(x)
            
            out = np.concatenate((peeky_h,out), axis=2)
            out = self.lstm.forward(out)
            out = np.concatenate((peeky_h,out), axis=2)
            score = self.affine.forward(out)
            
            char_id = np.argmax(score.flatten())
            sampled.append(char_id)
            
        return sampled

class PeekySeq2seq(Seq2seq):
    # 구현한 seq2seq 클래스를 계승, 초기화부분만을 변경
    def __init__(self, vocab_size, wordvec_size, hidden_size):
        V, D, H = vocab_size, wordvec_size, hidden_size
        self.encoder = Encoder(V,D,H)
        self.decoder = PeekyDecoder(V,D,H)
        self.softmax = TimeSoftmaxWithLoss()
        
        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

