#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class TimeLSTM:
    def __init__(self, Wx, Wh, b, stateful = False):
        self.params = [Wx, Wh, b] # 파라미터 저장
        self.grads = [np.zeros_like(Wx), np.zeros_like(Wh), np.zeros_like(b)] # grads
        self.layers = None
        
        self.h, self.c = None, None
        self.dh = None
        self.stateful = stateful
        
    def forward(self, xs):
        Wx, Wh, b = self.params
        N, T, D = xs.shape # batch size, time size, feature size
        H = Wh.shape[0] # 새로 만들 feature size
        
        self.layers = []
        hs = np.empty((N,T,H), dtype="f") # 각 time의 hs를 만듦(다음 affine 층으로 보내기 위해 필용)
        
        if not self.stateful or self.h is None: # 상태 유지 X or 아예 처음(self.h가 없다면)
            self.h = np.zeros((N,H), dtype="f") # 넘어오는 self.h의 원소를 모두 0으로 설정
        if not self.stateful or self.c is None:
            self.c = np.zeros((N,H), dtype="f") # memory cell과 hidden state의 크기는 같음
            
        for t in range(T):
            layer = LSTM(*self.params) # 각 time마다의 가중치는 모두 동일
            self.h, self.c = layer.forward(xs[:,t,:],self.h, self.c) # hidden state, memory cell 갱신
            hs[:,t,:] = self.h # 각 time의 hidden state 저장
            
            self.layers.append(layer) # layer을 쌓아올림
            
        return hs 
    
    def backward(self, dhs):
        Wx, Wh, b = self.params
        N, T, H = dhs.shape
        D = Wx.shape[0]
        
        dxs = np.empty((N,T,D), dtype="f")
        dh, dc = 0, 0 # 역전파는 적당한 길이로 끊기 때문
        
        grads = [0,0,0]
        for t in reversed(range(T)):
            layer = self.layers[t]
            # 다음 layer층에서 올라온 dhs. 다음 time에서 온 dh
            dx, dh, dc = layer.backward(dhs[:,t,:] + dh, dc)
            dxs[:,t,:] = dx
            # 모든 time에서의 grad를 한 곳에 합침
            for i, grad in enumerate(layer.grads):
                grads[i] += grad
        
        for i, grad in enumerate(grads):
            self.grads[i][...] = grad # grad를 저장
        self.dh = dh # 이전 timelstm에서 사용될 수 있기 때문에 저장해둠
        return dxs # 이전 layer 층을 위해 반환
    
    def set_state(self,h,c=None):
        self.h, self.c = h,c
        
    def reset_state(self):
        self.h, self.c = None, None

