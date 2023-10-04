import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
from matplotlib import cm
import scipy.io
from scipy.special import gamma



# 模型搭建
class Net(nn.Module):
    def __init__(self, num_hidden_layers, input_size, hidden_size):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layers = nn.ModuleList()
        for i in range(num_hidden_layers):
            self.hidden_layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out = torch.mul(self.input_layer(x), torch.tanh(self.input_layer(x)))
        for hidden_layer in self.hidden_layers:
            #out = torch.mul(hidden_layer(out), torch.tanh(hidden_layer(out)))
            out = torch.tanh(hidden_layer(out))
        out_final = self.output_layer(out)
        return out_final

def aaa(l,alpha):
    output=(l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output

def fpde(x, net , M , N, tau):

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

    uuu=torch.mul(torch.mul(u,(1-u)),(u-r))
    size_uuu = uuu.shape[0]
    uuu=uuu.reshape(size_uuu,1)
    return D_t - u_xx - uuu # 公式（1）


net = Net(num_hidden_layers=8, input_size=2, hidden_size=30)
mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.001 )
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # 选定调整方法



# 初始化 常量
ita = 0.001
delta = 4
r=0.75
rho = np.sqrt((4 * (1 + delta)))
sigma = delta * rho / (4 * (1 + delta))
M=30
N=20
alpha=0.99

boundnum=50
t_bc_zeros = np.zeros((boundnum, 1))
t_bc_ones = np.ones((boundnum, 1))
x_in_pos_one = 10*np.ones((boundnum, 1))
x_in_neg_one = -10*np.ones((boundnum, 1))
u_in_zeros = np.zeros((boundnum, 1))
u_in_ones = np.ones((boundnum, 1))


t = np.linspace(0, 1, N+1)
x = np.linspace(-10, 10, M+1)
tau=t[2]-t[1]
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x_collocation1 = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)
# u_exact =  torch.mul(torch.mul(torch.mul(pt_t_collocation,torch.mul(pt_t_collocation,pt_t_collocation)),(1-pt_x_collocation)),torch.sin(pt_x_collocation))
f = np.zeros((x.shape[0], 1))
Exact1 = np.zeros((x.shape[0], 1))
for i in range(x.shape[0]):
    f[i, 0] =0
    Exact1[i, 0] = 1 / (1 + np.exp((-1 / np.sqrt(2)) * (x[i, 0] + (1 - 2 * r) / np.sqrt(2) * t[i, 0])))
pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)


iterations = 5000
err=0
for epoch in range(iterations):
    optimizer.zero_grad()  # 梯度归0


    t_in_var = np.random.uniform(low=0, high=1.0, size=(boundnum, 1))
    x_bc_var = np.random.uniform(low=-10, high=10, size=(boundnum, 1))
    u_ic_t0 = 1 / (1 + np.exp(-x_bc_var / np.sqrt(2)))
    u_ic_t1 = 1 / (1 + np.exp((-1 / np.sqrt(2)) * (x_bc_var + (1 - 2 * r) / np.sqrt(2) )))
    # u_bc_0t = (ita/2+ita/2*np.tanh(sigma*ita*((((1+delta-ita)*rho)/(2*(1+delta)))*t_in_var)))**(1/delta)
    # u_bc_1t = (ita/2+ita/2*np.tanh(sigma*ita*(1+(((1+delta-ita)*rho)/(2*(1+delta)))*t_in_var)))**(1/delta)

    # 将数据转化为torch可用的
    pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)
    pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    pt_t_bc_ones = Variable(torch.from_numpy(t_bc_ones).float(), requires_grad=False)
    pt_u_ic_t0 = Variable(torch.from_numpy(u_ic_t0).float(), requires_grad=False)
    pt_u_ic_t1 = Variable(torch.from_numpy(u_ic_t1).float(), requires_grad=False)
    # pt_u_bc_0t = Variable(torch.from_numpy(u_bc_0t).float(), requires_grad=False)
    # pt_u_bc_1t = Variable(torch.from_numpy(u_bc_1t).float(), requires_grad=False)
    pt_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False)
    pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)
    pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)
    pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False)
    pt_u_in_ones = Variable(torch.from_numpy(u_in_ones).float(), requires_grad=False)

    # 求边界条件的损失
    net_bc_out = net(torch.cat([pt_t_bc_zeros, pt_x_bc_var], 1))  # u(x,t)的输出
    mse_u_2 = mse_cost_function1(net_bc_out, pt_u_ic_t0 )  # e = u(x,t)-(-sin(pi*x))  公式（2）

    net_bc_out = net(torch.cat([pt_t_bc_ones, pt_x_bc_var], 1))  # u(x,t)的输出
    mse_u_5 = mse_cost_function1(net_bc_out, pt_u_ic_t1)  # e = u(x,t)-(-sin(pi*x))  公式（2）

    net_bc_inr = net(torch.cat([pt_t_in_var, pt_x_in_pos_one], 1))  # 0=u(t,1) 公式（3)
    net_bc_inl = net(torch.cat([pt_t_in_var, pt_x_in_neg_one], 1))  # 0=u(t,0) 公式（4）

    mse_u_3 = mse_cost_function1(net_bc_inr, pt_u_in_ones)  # e = 0-u(t,10) 公式(3)
    mse_u_4 = mse_cost_function1(net_bc_inl, pt_u_in_zeros)  # e = 0-u(t,-10) 公式（4）


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





    minmse=np.min([mse_f_1.data,mse_u_4.data,mse_u_3.data,mse_u_2.data])
    w1=mse_f_1/minmse
    w2 = mse_u_2 / minmse
    w3 = mse_u_3 / minmse
    w4 = mse_u_4 / minmse
    w5 = mse_u_5 / minmse
    # 将误差(损失)累加起来
    loss = w1*mse_f_1+(w2*mse_u_2+w3*mse_u_3+w4*mse_u_4)
    MSE = mse_u_1
    #u_error_max = mse_u_1111
    #loss = 0.5*mse_f_1 + 0.5*(mse_u_1+mse_u_2)
    #np.savetxt('loss500.txt', (loss))

    loss.backward()  # 反向传播
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(epoch, "L2", error_L2)
            print(epoch, "MSE", MSE.data)
            print(epoch, "error max:", error_max)
            print(epoch, "error_mean", error_mean)
            if (error_L2 - err)<0.005 or err == 0 or epoch<2000:
                err = error_L2
            else:
                break



## 画图 ##


test_M=100
test_N=100
x0 = np.linspace(-10, 10, test_M)
t0 = np.linspace(0, 1, test_N)
#u_real=t**3*(1-x)*np.sin(x)

ms_t, ms_x = np.meshgrid(t0, x0)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
unn_torch = net(torch.cat([ pt_t,pt_x], 1))
unn_numpy = unn_torch.data.cpu().numpy()

u_realfla = np.zeros((x.shape[0], 1))
for i in range(x.shape[0]):
    u_realfla[i, 0]= 1 / (1 + np.exp((-1 / np.sqrt(2)) * (x[i, 0] + (1 - 2 * r) / np.sqrt(2) * t[i, 0])))
u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
u_real_numpy=u_real_torch.data.cpu().numpy()

mse_torch = mse_cost_function1(u_real_torch, unn_torch)
mse_numpy = mse_torch.data.cpu().numpy()

error= u_real_numpy-unn_numpy
error_mean = np.mean(np.abs(error))

unn_matrix = unn_numpy.reshape(test_M, test_N)
u_real_matrix = u_real_numpy.reshape(test_M, test_N)
error_matrix= error.reshape(test_M, test_N)


print("error max:",(np.abs(error)).max())
print("error mean:",error_mean)
error_L2=np.linalg.norm(error,ord=2)/np.linalg.norm(u_real_numpy,ord=2)
print("error L2:",error_L2)
print("error mse:",mse_numpy)
np.savetxt('error09.txt',(error))
np.savetxt('u_real09.txt',(u_real_numpy))
np.savetxt('unn09.txt',(unn_numpy))
#f = open("error.txt", 'a')
#f.write(error)
#f.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(ms_t, ms_x, unn_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.savefig('udata.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, error_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('NE')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='z')
plt.savefig('error.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, u_real_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0001, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

plt.savefig('u_real.png')
plt.close(fig)



'''
test_M = 100
test_N = 100
x0 = np.linspace(0.01, 0.99, test_M)
t0 = np.linspace(0, 1, test_N)
# u_real=t**3*(1-x)*np.sin(x)

ms_t, ms_x = np.meshgrid(t0, x0)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
unn_torch = net(torch.cat([pt_t, pt_x], 1))
unn_numpy = unn_torch.data.cpu().numpy()

u_realfla = np.zeros((x.shape[0], 1))
for i in range(x.shape[0]):
    u_realfla[i, 0] = t[i, 0] ** alpha * np.sin(np.pi * x[i, 0])
u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
u_real_numpy = u_real_torch.data.cpu().numpy()

mse_torch = mse_cost_function1(u_real_torch, unn_torch)
mse_numpy = mse_torch.data.cpu().numpy()

error =  u_real_numpy-unn_numpy
error_mean = np.mean(np.abs(error))

unn_matrix = unn_numpy.reshape(test_M, test_N)
u_real_matrix = u_real_numpy.reshape(test_M, test_N)
error_matrix = error.reshape(test_M, test_N)

print("error max:", np.abs(error).max())
print("error mean:", error_mean)
error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(u_real_numpy, ord=2)
print("error L2:", error_L2)
print("error mse:", mse_numpy)
np.savetxt('error0-1.txt', (error))
np.savetxt('u_real0-1.txt', (u_real_numpy))
np.savetxt('pt_u00-1.txt', (unn_numpy))
# f = open("error.txt", 'a')
# f.write(error)
# f.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(ms_t, ms_x, unn_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.savefig('udata_fpde0-1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, error_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('error')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='both')
plt.savefig('errordata_fpde0-1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, u_real_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

plt.savefig('eu_real_fpde0-1.png')
plt.close(fig)
'''
