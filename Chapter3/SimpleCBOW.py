import sys
import os
sys.path.append(".")
import numpy as np
from base_layer import MatMul,SoftmaxWithLoss

class SimpleCBOW:
    def __init__(self, vocab_size, hidden_size):
        V, H = vocab_size, hidden_size
        
        # 가중치 초기화
        W_in = 0.01 * np.random.randn(V,H).astype("f")
        W_out = 0.01 * np.random.randn(H,V).astype("f") # 32비트 부동소수점 수
        
        # 계층 생성
        # input layers
        self.in_layer0 = MatMul(W_in)
        self.in_layer1 = MatMul(W_in)
        # output layer
        self.out_layer = MatMul(W_out)
        # loss layer
        self.loss_layer = SoftmaxWithLoss()
        
        # 모든 가중치와 기울기를 리스트에 모은다.
        layers = [self.in_layer0, self.in_layer1, self.out_layer]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads
        
        # 인스턴스 변수에 단어에 분산 표현을 저장
        self.words_vecs = W_in
        
    def forward(self, contexts, target):
        h0 = self.in_layer0.forward(contexts[:,0])
        h1 = self.in_layer1.forward(contexts[:,1])
        h = (h0+h1)*0.5
        score = self.out_layer.forward(h)
        loss = self.loss_layer.forward(target, score)
        return loss
        
    # grads 갱신
    def backward(self, dout=1):
        ds = self.loss_layer.backward(dout)
        da = self.out_layer.backward(ds)
        da *= 0.5
        self.in_layer1.backward(da)
        self.in_layer0.backward(da)
        return None