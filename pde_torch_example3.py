
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
from matplotlib import cm
import scipy.io
from scipy.special import gamma
import time
from scipy.linalg import hankel, qr, svd
import os
t0=time.time()

# 模型搭建
class Net(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size,activation):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)
        self.active = activation
    def forward(self, x):
        out = self.input_layer(x) *self.active( self.input_layer(x))
        for hidden_layer in self.hidden_layers:
            out=hidden_layer(out)*self.active(hidden_layer(out))
        out_NN = self.output_layer(out)
        xs=torch.mul(x[:, 0]**alpha,torch.sin(np.pi*x[:, 1]))
        out_final = torch.mul(xs,out_NN[:,0])
        size_out=out_final.shape[0]
        out_final=out_final.reshape(size_out,1)
        return out_final



def aaa(l,alpha):
    output=(l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output

def a_hat(l, alpha, tau):
    return tau ** (-alpha)/(1-alpha)


def fpde(x, net , M , N, tau):
#
    u = net(x)  # 网络得到的数据

    u_matrix = u.reshape(M+1, N+1)
    u_matrix = u_matrix.detach().numpy()
    D_t=np.zeros(((M+1,N+1)))

    for n in range(1,N+1):
        for i in range(1,M):
            D_t[i,n]=D_t[i,n]+aaa(0,alpha)*tau**(-alpha)/gamma(2-alpha)*u_matrix[i][n]
            for k in range(1,n):
                D_t[i,n]=D_t[i,n]-((aaa(n-k-1,alpha)-aaa(n-k,alpha))*tau**(-alpha)/gamma(2-alpha)*u_matrix[i][k])
            D_t[i,n]=D_t[i,n]-aaa(n-1,alpha)*tau**(-alpha)/gamma(2-alpha)*u_matrix[i][0]

    D_t = D_t.flatten()[:,None]
    D_t = Variable(torch.from_numpy(D_t).float(), requires_grad=False)
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    d_t = u_tx[:, 0].unsqueeze(-1)
    d_x = u_tx[:, 1].unsqueeze(-1)
    u_xx = torch.autograd.grad(d_x, x, grad_outputs=torch.ones_like(d_x),
                               create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数

    #w = torch.tensor(0.01 / np.pi)

    uuu=torch.mul(u,(1-u))
    size_uuu = uuu.shape[0]
    uuu=uuu.reshape(size_uuu,1)
    return D_t - u_xx - uuu # 公式（1）

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = Net(num_hidden_layers=6, input_size=2, hidden_size=30,activation=torch.tanh).to(device)
mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)





# 初始化 常量

M=50
N=50
alpha=0.9

t = np.linspace(0.01, 0.99, N+1)
x = np.linspace(0.01, 0.99, M+1)
tau=t[2]-t[1]
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x_collocation1 = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)
Exact1 = t ** 2 *((x*(1-x))**1.5 )
f1=2 / gamma(3 - alpha) * (x - x ** 2) ** 1.5 * ((t) ** (2 - alpha))
f2=-(t ** 2) * (0.75 * (x - x ** 2) ** (-0.5) * (1 - 2 * x) ** 2 - 3 * (
                x - x ** 2) ** 0.5)
f3=-t**2 * ((x - x ** 2) ** 1.5) * (1 - t** 2 * ((x - x ** 2) ** 1.5)) 
f =f1+f2+f3


pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)


collection_train_time = []
collection_l2 = []
iterations = 1000
epoc=6
randomlist=[]
for i in range(epoc):
    for epoch in range(iterations):
        optimizer.zero_grad()  # 梯度归0

        # 将变量x,t带入公式（1）
        f_out = fpde(torch.cat([pt_t_collocation1,pt_x_collocation1], 1), net, M,N,tau)  # output of f(x,t) 公式（1）
        mse_f_1 = mse_cost_function1(f_out, pt_f_collocation1)
        net_u_in = net(torch.cat([pt_t_collocation1, pt_x_collocation1], 1))
        mse_u_1 = mse_cost_function1(net_u_in, pt_u_collocation1)
        error = net_u_in-pt_u_collocation1
        error = error.data.cpu().numpy()
        error_max = (np.abs(error)).max()
        error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(Exact1, ord=2)
        error_mean = np.mean(np.abs(error))


        # 将误差(损失)累加起来
        loss = mse_f_1
        MSE = mse_u_1


        loss.backward()  # 反向传播
        optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta
        t1 = time.time()
        train_time = t1 - t0
        collection_train_time = np.append(collection_train_time, train_time)
        collection_l2=np.append(collection_l2,error_L2)
        with torch.autograd.no_grad():
            if epoch % 50 == 0:
                print(epoch, "Traning Loss:", loss.data)
                print(epoch, "L2", error_L2)
                print(epoch, "MSE", MSE.data)
                print(epoch, "error max:", error_max)
                print(epoch, "error_mean", error_mean)
# torch.save(net, 'example3_1.pkl')
    np.savetxt('time_fPINNs_50'+str(i+1)+'.txt', (collection_train_time))
    np.savetxt('l2_fPINNs2_50'+str(i+1)+'.txt', (collection_l2))


## 画图 ##


# test_M=100
# test_N=100
# x0 = np.linspace(0, 1, test_M)
# t0 = np.linspace(0, 1, test_N)
# #u_real=t**3*(1-x)*np.sin(x)

# ms_t, ms_x = np.meshgrid(t0, x0)
# x = np.ravel(ms_x).reshape(-1, 1)
# t = np.ravel(ms_t).reshape(-1, 1)
# pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
# pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
# unn_torch = net(torch.cat([ pt_t,pt_x], 1))
# unn_numpy = unn_torch.data.cpu().numpy()

# u_realfla = np.zeros((x.shape[0], 1))
# for i in range(x.shape[0]):
#      u_realfla[i, 0] = t[i, 0] ** 2*((x[i, 0] * (1 - x[i, 0])) ** 1.5)
# u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
# u_real_numpy=u_real_torch.data.cpu().numpy()

# mse_torch = mse_cost_function1(u_real_torch, unn_torch)
# mse_numpy = mse_torch.data.cpu().numpy()

# error= u_real_numpy-unn_numpy
# error_mean = np.mean(np.abs(error))

# unn_matrix = unn_numpy.reshape(test_M, test_N)
# u_real_matrix = u_real_numpy.reshape(test_M, test_N)
# error_matrix= error.reshape(test_M, test_N)


# print("error max:",(np.abs(error)).max())
# print("error mean:",error_mean)
# error_L2=np.linalg.norm(error,ord=2)/np.linalg.norm(u_real_numpy,ord=2)
# print("error L2:",error_L2)
# print("error mse:",mse_numpy)
# np.savetxt('error0.1-1.txt',(error))
# np.savetxt('u_real0.1-1.txt',(u_real_numpy))
# np.savetxt('pt_u00.1-1.txt',(unn_numpy))
# #f = open("error.txt", 'a')
# #f.write(error)
# #f.close()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(ms_t, ms_x, unn_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.savefig('udata_fpde0.1-1.png')
# plt.close(fig)

# fig = plt.figure()
# bx = fig.add_subplot(projection='3d')
# bx.plot_surface(ms_t, ms_x, error_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
# bx.set_xlabel('t')
# bx.set_ylabel('x')
# bx.set_zlabel('NE')
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
# plt.ticklabel_format(style='sci',scilimits=(0,0),axis='both')
# plt.savefig('errordata_fpde0.1-1.png')
# plt.close(fig)

# fig = plt.figure()
# bx = fig.add_subplot(projection='3d')
# bx.plot_surface(ms_t, ms_x, u_real_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
# bx.set_xlabel('t')
# bx.set_ylabel('x')
# bx.set_zlabel('u')
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率

# plt.savefig('eu_real_fpde0.1-1.png')
# plt.close(fig)



# '''
# test_M = 100
# test_N = 100
# x0 = np.linspace(0.01, 0.99, test_M)
# t0 = np.linspace(0, 1, test_N)
# # u_real=t**3*(1-x)*np.sin(x)

# ms_t, ms_x = np.meshgrid(t0, x0)
# x = np.ravel(ms_x).reshape(-1, 1)
# t = np.ravel(ms_t).reshape(-1, 1)
# pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
# pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
# unn_torch = net(torch.cat([pt_t, pt_x], 1))
# unn_numpy = unn_torch.data.cpu().numpy()

# u_realfla = np.zeros((x.shape[0], 1))
# for i in range(x.shape[0]):
#     u_realfla[i, 0] = t[i, 0] ** alpha * np.sin(np.pi * x[i, 0])
# u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
# u_real_numpy = u_real_torch.data.cpu().numpy()

# mse_torch = mse_cost_function1(u_real_torch, unn_torch)
# mse_numpy = mse_torch.data.cpu().numpy()

# error =  u_real_numpy-unn_numpy
# error_mean = np.mean(np.abs(error))

# unn_matrix = unn_numpy.reshape(test_M, test_N)
# u_real_matrix = u_real_numpy.reshape(test_M, test_N)
# error_matrix = error.reshape(test_M, test_N)

# print("error max:", np.abs(error).max())
# print("error mean:", error_mean)
# error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(u_real_numpy, ord=2)
# print("error L2:", error_L2)
# print("error mse:", mse_numpy)
# np.savetxt('error0-1.txt', (error))
# np.savetxt('u_real0-1.txt', (u_real_numpy))
# np.savetxt('pt_u00-1.txt', (unn_numpy))
# # f = open("error.txt", 'a')
# # f.write(error)
# # f.close()
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# ax.plot_surface(ms_t, ms_x, unn_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
# ax.set_xlabel('t')
# ax.set_ylabel('x')
# ax.set_zlabel('u')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.savefig('udata_fpde0-1.png')
# plt.close(fig)

# fig = plt.figure()
# bx = fig.add_subplot(projection='3d')
# bx.plot_surface(ms_t, ms_x, error_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
# bx.set_xlabel('t')
# bx.set_ylabel('x')
# bx.set_zlabel('error')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率
# plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
# plt.savefig('errordata_fpde0-1.png')
# plt.close(fig)

# fig = plt.figure()
# bx = fig.add_subplot(projection='3d')
# bx.plot_surface(ms_t, ms_x, u_real_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
# bx.set_xlabel('t')
# bx.set_ylabel('x')
# bx.set_zlabel('u')
# plt.rcParams['savefig.dpi'] = 300  # 图片像素
# plt.rcParams['figure.dpi'] = 300  # 分辨率

# plt.savefig('eu_real_fpde0-1.png')
# plt.close(fig)
# '''