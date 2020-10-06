import numpy as np

# 근사적으로 PPMI 구현
def ppmi(C, verbose = False, eps = 1e-8):
    M = np.zeros_like(C, dtype = np.float32) 
    N = np.sum(C) # 전체 표본 공간
    S = np.sum(C, axis=0) # 어차피 대칭행렬이기 때문에 indexing하기 쉽게 axis=0으로
    total = C.shape[0] * C.shape[1]
    cnt = 0
    
    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i,j]*N / (S[j]*S[i])+ eps)
            M[i,j] = max(0, pmi)
            
            if verbose: # 중간중간 진행상황을 알려줌
                cnt += 1
                if cnt % (total//100)==0:
                    print("%.2f%% 완료" %(100*cnt/total)) # x%정도 진행함
                    
    return M