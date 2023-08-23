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



# 模型搭建
class Net(nn.Module):
    def __init__(self, NN): # NL n个l（线性，全连接）隐藏层， NN 输入数据的维数， 128 256
        # NL是有多少层隐藏层
        # NN是每层的神经元数量
        super(Net, self).__init__()

        self.input_layer = nn.Linear(2, NN)
        self.hidden_layer1 = nn.Linear(NN,int(NN)) ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.hidden_layer2 = nn.Linear(int(NN), int(NN))  ## 原文这里用NN，我这里用的下采样，经过实验验证，“等采样”更优
        self.hidden_layer3 = nn.Linear(int(NN), int(NN))
        self.hidden_layer4 = nn.Linear(int(NN), int(NN))
        self.hidden_layer5 = nn.Linear(int(NN), int(NN))
        self.hidden_layer6 = nn.Linear(int(NN), int(NN))
        self.hidden_layer7 = nn.Linear(int(NN), int(NN))
        self.hidden_layer8 = nn.Linear(int(NN), int(NN))
        self.output_layer = nn.Linear(int(NN), 1)

    def forward(self, x): # 一种特殊的方法 __call__() 回调
        #out = torch.tanh(self.input_layer(x))
        #out = torch.tanh(self.hidden_layer1(out))
        #out = torch.tanh(self.hidden_layer2(out))
        #out = torch.tanh(self.hidden_layer3(out))
        #out = torch.tanh(self.hidden_layer4(out))
        #out = torch.tanh(self.hidden_layer5(out))
        #out = torch.tanh(self.hidden_layer6(out))
        #out = torch.tanh(self.hidden_layer7(out))
        #out = torch.tanh(self.hidden_layer8(out))

        out = torch.mul(self.input_layer(x),torch.tanh(self.input_layer(x)))
        out = torch.mul(self.hidden_layer2(out),torch.tanh(self.hidden_layer2(out)))
        out = torch.mul(self.hidden_layer3(out),torch.tanh(self.hidden_layer3(out)))
        out = torch.mul(self.hidden_layer4(out),torch.tanh(self.hidden_layer4(out)))
        out = torch.mul(self.hidden_layer5(out),torch.tanh(self.hidden_layer5(out)))
        out = torch.mul(self.hidden_layer6(out),torch.tanh(self.hidden_layer6(out)))
        out = torch.mul(self.hidden_layer7(out), torch.tanh(self.hidden_layer7(out)))
        out = torch.mul(self.hidden_layer8(out), torch.tanh(self.hidden_layer8(out)))
        out_NN = self.output_layer(out)
        xs=torch.mul(x[:, 0],torch.sin(np.pi*x[:, 1]))
        out_final = torch.mul(xs,out_NN[:,0])
        size_out=out_final.shape[0]
        out_final=out_final.reshape(size_out,1)
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

    uuu=torch.mul(torch.mul(u,(1-u)),(u-0.5))
    size_uuu = uuu.shape[0]
    uuu=uuu.reshape(size_uuu,1)
    return D_t - u_xx - uuu # 公式（1）


net = Net(30)
mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.001 )
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # 选定调整方法



# 初始化 常量

M=30
N=30
alpha=0.6

t = np.linspace(0.00001, 1, N+1)
x = np.linspace(0, 1, M+1)
tau=t[2]-t[1]
ms_t, ms_x = np.meshgrid(t, x)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x_collocation1 = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)
# u_exact =  torch.mul(torch.mul(torch.mul(pt_t_collocation,torch.mul(pt_t_collocation,pt_t_collocation)),(1-pt_x_collocation)),torch.sin(pt_x_collocation))
f = np.zeros((x.shape[0], 1))
Exact1 = np.zeros((x.shape[0], 1))
f_value = np.sin(np.pi * x) * (gamma(1 + alpha) + (np.pi ** 2) * (t ** alpha) - t ** alpha * (1 - t ** np.sin(np.pi * x)) * (t ** alpha * np.sin(np.pi * x) - 0.5))
r=0.1
sigma_f = r * f_value
noise = np.random.normal(0, sigma_f)
f = np.sin(np.pi*x)*(gamma(1+alpha)+(np.pi**2)*(t**alpha)-t**alpha*(1-t**np.sin(np.pi*x))*(t**alpha*np.sin(np.pi*x)-0.5))+noise
Exact1 = t ** alpha * np.sin(np.pi*x)
pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)


iterations = 2000
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
    #u_error_max = mse_u_1111
    #loss = 0.5*mse_f_1 + 0.5*(mse_u_1+mse_u_2)
    #np.savetxt('loss500.txt', (loss))

    loss.backward()  # 反向传播
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 50 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(epoch, "L2", error_L2)
            print(epoch, "MSE", MSE.data)
            print(epoch, "error max:", error_max)
            print(epoch, "error_mean", error_mean)



## 画图 ##


test_M=100
test_N=100
x0 = np.linspace(0, 1, test_M)
t0 = np.linspace(0.000000001, 1, test_N)
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
     u_realfla[i, 0] = t[i, 0] ** alpha * np.sin(np.pi*x[i, 0])
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
np.savetxt('03error.txt',(error))
np.savetxt('03ureal.txt',(u_real_numpy))
np.savetxt('03unn.txt',(unn_numpy))
#f = open("error.txt", 'a')
#f.write(error)
#f.close()
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(ms_t, ms_x, unn_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.savefig('udata_fpde0.1-1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, error_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('NE')
bx.view_init(elev=10., azim=140)
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='z')
plt.savefig('errordata_fpde0.1-1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, u_real_matrix, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率

plt.savefig('eu_real_fpde0.1-1.png')
plt.close(fig)

