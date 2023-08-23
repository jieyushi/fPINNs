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
import time


t0=time.time()
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
            out = torch.mul(hidden_layer(out), torch.tanh(hidden_layer(out)))
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

    uuu=torch.mul(torch.mul(u,(1-u)),(u-1))
    size_uuu = uuu.shape[0]
    uuu=uuu.reshape(size_uuu,1)
    return D_t - u_xx - uuu # 公式（1）


net = Net(num_hidden_layers=8, input_size=2, hidden_size=20)
mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

#optimizer = torch.optim.SGD(net.parameters(), lr=0.001 )
#scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # 选定调整方法



# 初始化 常量
#nbound=20
#t_bc_zeros = np.zeros((nbound, 1))
#x_in_pos_one = np.ones((nbound, 1))
#x_in_neg_one = np.zeros((nbound, 1))
#u_in_zeros = np.zeros((nbound, 1))
M=20
N=20
alpha=0.9
t = np.linspace(0, 1, N+1)
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
for i in range(x.shape[0]):
    f[i, 0] = 6 / gamma(4 - alpha) * t[i, 0] ** (3 - alpha) * (1 - x[i, 0]) * np.sin(x[i, 0]) + 2 * t[i, 0] ** 3 * np.cos(x[i, 0]) + t[i, 0] ** 3 * (1 - x[i, 0]) * np.sin(x[i, 0]) - t[i, 0] ** 3 * (1 - x[i, 0]) * np.sin(x[i, 0]) * (1 - t[i, 0] ** 3 * (1 - x[i, 0])) * (t[i, 0] ** 3 * (1 - x[i, 0]) * np.sin(x[i, 0]) - 1)
    Exact1[i, 0] = t[i, 0] ** 3 * (1 - x[i, 0]) * np.sin(x[i, 0])
pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)

collection_l2=[]
collection_time=[]
collection_error_mean=[]
collection_error_max=[]
collection_l2=[]
collection_loss=[]

iterations = 3000
for epoch in range(iterations):
    optimizer.zero_grad()  # 梯度归0

    # 求边界条件的误差
    # 初始化变量
    #t_in_var = np.random.uniform(low=0, high=1.0, size=(nbound, 1))
    #x_bc_var = np.random.uniform(low=0, high=1.0, size=(nbound, 1))
    #u_bc_zero = np.zeros((nbound, 1))

    # 将数据转化为torch可用的
    #pt_x_bc_var = Variable(torch.from_numpy(x_bc_var).float(), requires_grad=False)
    #pt_t_bc_zeros = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    #pt_u_bc_zeros = Variable(torch.from_numpy(u_bc_zero).float(), requires_grad=False)
    #pt_x_in_pos_one = Variable(torch.from_numpy(x_in_pos_one).float(), requires_grad=False)
    #pt_x_in_neg_one = Variable(torch.from_numpy(x_in_neg_one).float(), requires_grad=False)
    #pt_t_in_var = Variable(torch.from_numpy(t_in_var).float(), requires_grad=False)
    #pt_u_in_zeros = Variable(torch.from_numpy(u_in_zeros).float(), requires_grad=False)

    # 求边界条件的损失
    #net_bc_out = net(torch.cat([pt_t_bc_zeros,pt_x_bc_var], 1))  # u(x,t)的输出
    #mse_u_2 = mse_cost_function(net_bc_out, pt_u_bc_zeros)  # e = u(x,t)-(-sin(pi*x))  公式（2）

    #net_bc_inr = net(torch.cat([ pt_t_in_var,pt_x_in_pos_one], 1))  # 0=u(t,1) 公式（3)
    #net_bc_inl = net(torch.cat([ pt_t_in_var,pt_x_in_neg_one], 1))  # 0=u(t,0) 公式（4）

    #mse_u_3 = mse_cost_function(net_bc_inr, pt_u_in_zeros)  # e = 0-u(t,1) 公式(3)
    #mse_u_4 = mse_cost_function(net_bc_inl, pt_u_in_zeros)  # e = 0-u(t,-1) 公式（4）

    # 求PDE函数式的误差
    # 初始化变量
    #x_collocation = np.random.uniform(low=-1.0, high=1.0, size=(2000, 1))
    #t_collocation = np.random.uniform(low=0.0, high=1.0, size=(2000, 1))

    
    # 将变量x,t带入公式（1）
    f_out = fpde(torch.cat([pt_t_collocation1,pt_x_collocation1], 1), net, M,N,tau)  # output of f(x,t) 公式（1）
    mse_f_1 = mse_cost_function1(f_out, pt_f_collocation1)
    net_u_in = net(torch.cat([pt_t_collocation1, pt_x_collocation1], 1))
    mse_u_1 = mse_cost_function1(net_u_in, pt_u_collocation1)
    error = net_u_in-pt_u_collocation1
    error = error.data.cpu().numpy()
    error_max = (np.abs(error)).max()
    error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(Exact1, ord=2)
    t1 = time.time()
    train_time = t1 - t0
    collection_time = np.append(collection_time, train_time)
    error_mean = np.mean(np.abs(error))
    collection_l2=np.append(collection_l2,error_L2)
    collection_error_mean = np.append(collection_error_mean, error_mean)
    collection_error_max = np.append(collection_error_max, error_max)


    # 将误差(损失)累加起来
    loss = mse_f_1
    MSE = mse_u_1
    mse= MSE.data.cpu().numpy()
    collection_loss=np.append(collection_loss,loss.data)
    #u_error_max = mse_u_1111
    #loss = 0.5*mse_f_1 + 0.5*(mse_u_1+mse_u_2)
    #np.savetxt('loss500.txt', (loss))

    loss.backward()  # 反向传播
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 99 == 0:
            print(epoch,"Traning Loss:", loss.data)
            print(error_L2)
            print(mse)
t1=time.time()
train_time=t1-t0
print(train_time)

plt.plot(collection_l2)
plt.plot(collection_error_max)
plt.plot(collection_error_mean)
plt.show()
# plt.rcParams['savefig.dpi'] = 300 #图片像素
# plt.rcParams['figure.dpi'] = 300 #分辨率
np.savetxt('l2_fpinn_5.txt',(collection_l2))
np.savetxt('time_fpinn_5.txt',(collection_time))
#np.savetxt('collection_error_max.txt',(collection_error_max))
#np.savetxt('collection_error_mean.txt',(collection_error_mean))
#np.savetxt('loss_1e-5-4.txt',(collection_loss))


print(error_L2)
print(mse)




## 画图 ##


def u_exact(x, t):
    output=t ** 3 * (1 - x) * np.sin(x)
    return output
test_M=100
test_N=100
x0 = np.linspace(0, 1, test_M)
t0 = np.linspace(0, 1, test_N)
#u_real=t**3*(1-x)*np.sin(x)

ms_t, ms_x = np.meshgrid(t0, x0)
x = np.ravel(ms_x).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x = Variable(torch.from_numpy(x).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
pt_u0 = net(torch.cat([ pt_t,pt_x], 1))

u_real0=torch.mul(torch.mul(torch.mul(pt_t,pt_t),pt_t), torch.mul((1 - pt_x) , torch.sin(pt_x)))
mse_utest = mse_cost_function1(u_real0, pt_u0)
mse_utest = mse_utest.data.cpu().numpy()
u = pt_u0.data.cpu().numpy()
pt_u00 = u.reshape(test_M, test_N)
u_real=np.zeros((pt_u00.shape))
for n in range(test_N):
   for m in range(test_M):
       u_real[m,n]=u_exact(x0[m],t0[n])
error=u_real-pt_u00
abserror=np.abs(u_real-pt_u00)
print("error_max",abserror.max())
print("error_min",abserror.min())
print("error_mean",abserror.mean())
error_L2=np.linalg.norm(error,ord=2)/np.linalg.norm(u_real,ord=2)
print("error_L2",error_L2)
print(mse_utest)
#np.savetxt('error_0.1.txt',(error))
#np.savetxt('u_real_0.1.txt',(u_real))
#np.savetxt('unn_0.1.txt',(pt_u00))
#f = open("error.txt", 'a')
#f.write(error)
#f.close()

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.set_zlim([0, 0.25])
ax.plot_surface(ms_t, ms_x, pt_u00, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('t')
ax.set_ylabel('x')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.savefig('udata_fpde_0.1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, error, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('NE')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.ticklabel_format(style='sci',scilimits=(0,0),axis='z')
plt.savefig('errordata_fpde_0.1.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(ms_t, ms_x, u_real, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('t')
bx.set_ylabel('x')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率

plt.savefig('u_real_fpde_0.1.png')
plt.close(fig)
