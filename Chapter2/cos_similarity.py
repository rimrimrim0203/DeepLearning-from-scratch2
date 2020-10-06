import numpy as np
def cos_similarity(x,y, eps=1e-8):
    # 만약 제로 벡터이면 eps 유지 -> 0으로 나누는 것을 막아줌 
    # 만약 제로 벡터가 아니면 eps이 반올림되어 다른 값에 흡수 -> 대부분 최종 결과에 영향 X
    nx = x/np.sqrt(np.sum(x**2)+eps)
    ny = y/np.sqrt(np.sum(y**2)+eps)
    
    return np.dot(nx, ny)