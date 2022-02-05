import numpy as np
import math
import cmath
from matplotlib import colors, pyplot as plt
def u(n):
    return 1 if n>= 0 else 0

def x(N, n, T, f):
    return cmath.sin(2*cmath.pi*f*n*T)*(u(n)-u(n-N))

def W(N, k):
    return cmath.exp(-1j*2*cmath.pi*k/N)

def bit_reverse(in_i, M):
    out = 0
    for i in range(M):
        out += (in_i%2)*(2**(M-i-1))
        in_i = int(in_i/2)
    return out

def FFT(x_list, N):
    rounds = int(math.log(N,2))
    x_list = [x_list[bit_reverse(i, rounds)] for i in range(N)]
    print(x_list)
    Ws = []
    for i in range(2**rounds):
        Ws.append(W(N, i))
    for round in range(rounds):
        temp = x_list
        round += 1
        for k in range(N // (2 ** round)):
        # 每轮分了N // (2 ** round)组
            for j in range(2 ** (round-1)):
            # 每轮每组人数的一半需改变
                X1 = x_list[j + k * (2 ** round)]
                X2 = x_list[j + k * (2 ** round) + 2 ** (round - 1)] * Ws[j * 2 ** (rounds - round)]
                temp[j + k * (2 ** round)] = X1 + X2
                temp[j + k * (2 ** round) + 2 ** (round - 1)] = X1 - X2
        x_list = temp
    return x_list

fs = [50, 50, 100, 1000]
Ns = [32, 64, 32, 32]
Ts = [0.005, 0.005, 0.005, 0.0012]
for i in range(4):
    x_list = [ x(Ns[i], k, Ts[i], fs[i]) for k in range(Ns[i])]
    x_list = FFT(x_list, Ns[i])
    # print(x_list)
    result = np.array([abs(i) for i in x_list])
    max, min = np.max(result), np.min(result)
    result = [(x-min)/(max-min) for x in result]
    # print(result)
    fig ,ax = plt.subplots()
    ax.stem(range(int(Ns[i]/2)),result[:int(Ns[i]/2)])
    ax.set_title('task'+str(i+1))
    ax.set_xlabel('k')
    ax.set_ylabel('X(k)')
    plt.savefig('task'+str(i+1))
    plt.close()

# x_list = np.zeros(64,dtype=np.complex64)
# for i in range(32):
#     x_list[i] = x(64, i, Ts[3], fs[3]) 
# x_list = FFT(x_list, 64)
# print(x_list)
# result = np.array([abs(i) for i in x_list])
# max, min = np.max(result), np.min(result)
# result = [(x-min)/(max-min) for x in result]
# print(result)
# fig ,ax = plt.subplots()
# ax.stem(range(32),result[:32])
# ax.set_title('task'+str(5))
# ax.set_xlabel('k')
# ax.set_ylabel('X(k)')
# plt.savefig('task'+str(5))
# plt.show()
# plt.close()
