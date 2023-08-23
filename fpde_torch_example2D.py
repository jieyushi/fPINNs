# soft constrain
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
import time

t0 = time.time()


# 模型搭建
class Net(nn.Module):
    def __init__(self, NN):
        super(Net, self).__init__()

        self.input_layer = nn.Linear(3, NN)
        self.hidden_layers = nn.ModuleList([nn.Linear(NN, NN) for _ in range(7)])  # Create a list of hidden layers
        self.output_layer = nn.Linear(NN, 1)

    def forward(self, x):
        out = torch.mul(self.input_layer(x), torch.tanh(self.input_layer(x)))

        for hidden_layer in self.hidden_layers:
            out = torch.mul(hidden_layer(out), torch.tanh(hidden_layer(out)))

        out_NN = self.output_layer(out)
        # xs = torch.mul(torch.mul(x[:, 0], torch.sin(np.pi * x[:, 1])), torch.sin(np.pi * x[:, 2]))
        # xs = torch.mul(torch.sin(np.pi * x[:, 1]), torch.sin(np.pi * x[:, 2]))

        # out_final = torch.mul(xs, out_NN[:, 0])
        # out_final = out_final.view(out_final.size(0), 1)  # Reshape the output tensor
        return out_NN


def aaa(l, alpha):
    output = (l + 1) ** (1 - alpha) - l ** (1 - alpha)
    return output


def fpde(x, net, M1, M2, N, tau):
    u = net(x)  # 网络得到的数据

    u_matrix = u.reshape(M1 + 1, M2 + 1, N + 1)
    u_matrix = u_matrix.detach().numpy()
    D_t = np.zeros(((M1 + 1, M2 + 1, N + 1)))

    for n in range(1, N + 1):
        for i1 in range(1, M1):
            for i2 in range(1, M2):
                D_t[i1, i2, n] = D_t[i1, i2, n] + aaa(0, alpha) * tau ** (-alpha) / gamma(2 - alpha) * u_matrix[i1][i2][
                    n]
                for k in range(1, n):
                    D_t[i1, i2, n] = D_t[i1, i2, n] - (
                                (aaa(n - k - 1, alpha) - aaa(n - k, alpha)) * tau ** (-alpha) / gamma(2 - alpha) *
                                u_matrix[i1][i2][k])
                D_t[i1, i2, n] = D_t[i1, i2, n] - aaa(n - 1, alpha) * tau ** (-alpha) / gamma(2 - alpha) * \
                                 u_matrix[i1][i2][0]
    D_t = D_t.flatten()[:, None]
    D_t = Variable(torch.from_numpy(D_t).float(), requires_grad=False)
    u_tx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(net(x)),
                               create_graph=True, allow_unused=True)[0]  # 求偏导数
    d_t = u_tx[:, 0].unsqueeze(-1)
    d_x1 = u_tx[:, 1].unsqueeze(-1)
    d_x2 = u_tx[:, 2].unsqueeze(-1)
    u_xx1 = torch.autograd.grad(d_x1, x, grad_outputs=torch.ones_like(d_x1),
                                create_graph=True, allow_unused=True)[0][:, 1].unsqueeze(-1)  # 求偏导数
    u_xx2 = torch.autograd.grad(d_x2, x, grad_outputs=torch.ones_like(d_x2),
                                create_graph=True, allow_unused=True)[0][:, 2].unsqueeze(-1)  # 求偏导数
    # w = torch.tensor(0.01 / np.pi)

    uuu = torch.mul(torch.mul(u, (1 - u)), (u - 1))
    size_uuu = uuu.shape[0]
    uuu = uuu.reshape(size_uuu, 1)
    return D_t - u_xx1 - u_xx2 - uuu  # 公式（1）


net = Net(30)
mse_cost_function1 = torch.nn.MSELoss(reduction='mean')  # Mean squared error
mse_cost_function2 = torch.nn.MSELoss(reduction='sum')  # Mean squared error
optimizer = torch.optim.Adam(net.parameters(), lr=2e-3)

# optimizer = torch.optim.SGD(net.parameters(), lr=0.001 )
# scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1) # 选定调整方法


# 初始化 常量
N = 20
M1 = 10
M2 = 10
alpha = 0.6

iterations = 20000

t = np.linspace(0, 1, N + 1)
x1 = np.linspace(0, 1, M1 + 1)
x2 = np.linspace(0, 1, M2 + 1)
tau = t[2] - t[1]
ms_t, ms_x1, ms_x2 = np.meshgrid(t, x1, x2)
x1 = np.ravel(ms_x1).reshape(-1, 1)
x2 = np.ravel(ms_x2).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)

pt_x1_collocation1 = Variable(torch.from_numpy(x1).float(), requires_grad=True)
pt_x2_collocation1 = Variable(torch.from_numpy(x2).float(), requires_grad=True)
pt_t_collocation1 = Variable(torch.from_numpy(t).float(), requires_grad=True)

boundnum=100

t_bc_zeros = np.zeros((boundnum, 1))
t_bc_ones = np.ones((boundnum, 1))
x_zero = np.zeros((boundnum, 1))
x_one = np.ones((boundnum, 1))

Exact1 = (1 + t ** 3) * (1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2)
f1 = 6 * t ** (3 - alpha) / gamma(4 - alpha) * (1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2)
f2 = 2 * (1 + t ** 3) * ((1 - x1) * np.sin(x1) * (1 - x2) * np.sin(x2))
f3 = 2 * (1 + t ** 3) * (np.cos(x1) * (1 - x2) * np.sin(x2) + (1 - x1) * np.sin(x1) * np.cos(x2))
f4 = -Exact1 * (1 - Exact1) * (Exact1 - 1)
f = f1 + f2 + f3 + f4

collection_l2 = []
collection_error_mean = []
collection_error_max = []
collection_l2 = []
collection_loss = []

for epoch in range(iterations):
    x1_bc_var = np.random.uniform(low=0, high=1, size=(boundnum, 1))
    x2_bc_var = np.random.uniform(low=0, high=1, size=(boundnum, 1))
    t0_ic_var = np.random.uniform(low=0, high=1, size=(boundnum, 1))


    pt_t_zero = Variable(torch.from_numpy(t_bc_zeros).float(), requires_grad=False)
    pt_t_one = Variable(torch.from_numpy(t_bc_ones).float(), requires_grad=False)
    pt_x_zero= Variable(torch.from_numpy(x_zero).float(), requires_grad=False)
    pt_x_one = Variable(torch.from_numpy(x_one).float(), requires_grad=False)

    pt_x1bc_collocation1 = Variable(torch.from_numpy(x1_bc_var).float(), requires_grad=True)
    pt_x2bc_collocation1 = Variable(torch.from_numpy(x2_bc_var).float(), requires_grad=True)
    pt_t0ic_collocation1 = Variable(torch.from_numpy(t0_ic_var).float(), requires_grad=True)

    # u_exact =  torch.mul(torch.mul(torch.mul(pt_t_collocation,torch.mul(pt_t_collocation,pt_t_collocation)),(1-pt_x_collocation)),torch.sin(pt_x_collocation))



    Exactt0 = (1 - x1_bc_var) * np.sin(x1_bc_var) * (1 - x2_bc_var) * np.sin(x2_bc_var)
    Exactt1 = 2 *(1 - x1_bc_var) * np.sin(x1_bc_var) * (1 - x2_bc_var) * np.sin(x2_bc_var)
    Exactbc_zero = t0_ic_var*0




    pt_f_collocation1 = Variable(torch.from_numpy(f).float(), requires_grad=True)
    pt_u_collocation1 = Variable(torch.from_numpy(Exact1).float(), requires_grad=True)
    pt_u_collocationt1 = Variable(torch.from_numpy(Exactt1).float(), requires_grad=True)
    pt_u_collocationt0 = Variable(torch.from_numpy(Exactt0).float(), requires_grad=True)
    pt_u_collocation_bczero = Variable(torch.from_numpy(Exactbc_zero).float(), requires_grad=True)

    optimizer.zero_grad()  # 梯度归0
    f_out = fpde(torch.cat([pt_t_collocation1, pt_x1_collocation1, pt_x2_collocation1], 1), net, M1, M2, N,
                 tau)  # output of f(x,t) 公式（1）
    mse_f_1 = mse_cost_function1(f_out, pt_f_collocation1)
    net_u_t0 = net(torch.cat([pt_t_zero, pt_x1bc_collocation1, pt_x2bc_collocation1], 1))
    net_u_t1 = net(torch.cat([pt_t_one, pt_x1bc_collocation1, pt_x2bc_collocation1], 1))
    net_u_x10 = net(torch.cat([pt_t0ic_collocation1, pt_x_zero, pt_x2bc_collocation1], 1))
    net_u_x11 = net(torch.cat([pt_t0ic_collocation1, pt_x_one, pt_x2bc_collocation1], 1))
    net_u_x20 = net(torch.cat([pt_t0ic_collocation1, pt_x1bc_collocation1, pt_x_zero], 1))
    net_u_x21 = net(torch.cat([pt_t0ic_collocation1, pt_x1bc_collocation1, pt_x_one], 1))
    net_u_in = net(torch.cat([pt_t_collocation1, pt_x1_collocation1, pt_x2_collocation1], 1))
    mse_u_t1 = mse_cost_function1(net_u_t1, pt_u_collocationt1)
    mse_u_t0 = mse_cost_function1(net_u_t0, pt_u_collocationt0)
    mse_u_x10 = mse_cost_function1(net_u_x10, pt_u_collocation_bczero)
    mse_u_x11 = mse_cost_function1(net_u_x11, pt_u_collocation_bczero)
    mse_u_x20 = mse_cost_function1(net_u_x20, pt_u_collocation_bczero)
    mse_u_x21 = mse_cost_function1(net_u_x21, pt_u_collocation_bczero)
    errort1 = net_u_in - pt_u_collocation1
    mse_u_1 = mse_cost_function1(net_u_in, pt_u_collocation1)
    error = net_u_in - pt_u_collocation1
    error = error.data.cpu().numpy()
    error_max = (np.abs(error)).max()
    error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(Exact1, ord=2)
    error_mean = np.mean(np.abs(error))
    collection_l2 = np.append(collection_l2, error_L2)
    collection_error_mean = np.append(collection_error_mean, error_mean)
    collection_error_max = np.append(collection_error_max, error_max)

    # 将误差(损失)累加起来
    # minmse=np.min([mse_f_1.data,mse_u_t0.data,mse_u_t1.data,mse_u_x10.data,mse_u_x11.data,mse_u_x20.data,mse_u_x21.data])
    minmse=np.min([mse_f_1.data,mse_u_t0.data,mse_u_x10.data,mse_u_x11.data,mse_u_x20.data,mse_u_x21.data])
    w1=mse_f_1/minmse
    w2 = mse_u_t0 / minmse
    w3 = mse_u_t1 / minmse
    w4 = mse_u_x10 / minmse
    w5 = mse_u_x11 / minmse
    w6 = mse_u_x20 / minmse
    w7 = mse_u_x21 / minmse
    # 将误差(损失)累加起来
    # loss = w1*mse_f_1+(w2*mse_u_t0+w3*mse_u_t1+w4*mse_u_x10+w5*mse_u_x11+w6*mse_u_x20+w7*mse_u_x21)
    loss = w1*mse_f_1+(w2*mse_u_t0+w4*mse_u_x10+w5*mse_u_x11+w6*mse_u_x20+w7*mse_u_x21)


    # loss = mse_f_1+mse_u_t0+mse_u_x10+mse_u_x11+mse_u_x20+mse_u_x21
    MSE = mse_u_1
    collection_loss = np.append(collection_loss, loss.data)
    # u_error_max = mse_u_1111
    # loss = 0.5*mse_f_1 + 0.5*(mse_u_1+mse_u_2)
    # np.savetxt('loss500.txt', (loss))

    loss.backward()  # 反向传播
    optimizer.step()  # This is equivalent to : theta_new = theta_old - alpha * derivative of J w.r.t theta

    with torch.autograd.no_grad():
        if epoch % 100 == 0:
            print(epoch, "Traning Loss:", loss.data)
            print(epoch, "L2", error_L2)
            print(epoch, "MSE", MSE.data)
            print(epoch, "error max:", error_max)
            print(epoch, "error_mean", error_mean)
t1 = time.time()
train_time = t1 - t0
print(train_time)

plt.plot(collection_l2)
plt.plot(collection_error_max)
plt.plot(collection_error_mean)
plt.show()
plt.rcParams['savefig.dpi'] = 300  # 图片像素
plt.rcParams['figure.dpi'] = 300  # 分辨率
np.savetxt('collection_l2.txt', (collection_l2))
np.savetxt('collection_error_max.txt', (collection_error_max))
np.savetxt('collection_error_mean.txt', (collection_error_mean))
np.savetxt('collection_loss5e-4.txt', (collection_loss))

test_M1 = 100
test_M2 = 100
test_N = 100
t0 = np.linspace(0, 1, test_N + 1)
x1 = np.linspace(0, 1, test_M1 + 1)
x2 = np.linspace(0, 1, test_M2 + 1)
# u_real=t**3*(1-x)*np.sin(x)
x1_plot, x2_plot = np.meshgrid(x1, x2)
ms_t, ms_x1, ms_x2 = np.meshgrid(t0, x1, x2)
x1 = np.ravel(ms_x1).reshape(-1, 1)
x2 = np.ravel(ms_x2).reshape(-1, 1)
t = np.ravel(ms_t).reshape(-1, 1)
pt_x1 = Variable(torch.from_numpy(x1).float(), requires_grad=True)
pt_x2 = Variable(torch.from_numpy(x2).float(), requires_grad=True)
pt_t = Variable(torch.from_numpy(t).float(), requires_grad=True)
unn_torch = net(torch.cat([pt_t, pt_x1, pt_x2], 1))
unn_numpy = unn_torch.data.cpu().numpy()

u_realfla = np.zeros((x1.shape[0], 1))
for i in range(x1.shape[0]):
    u_realfla[i, 0] = (1+t[i, 0] ** 3) * (1 - x1[i, 0]) * np.sin(x1[i, 0]) * (1 - x2[i, 0]) * np.sin(x2[i, 0])
u_real_torch = Variable(torch.from_numpy(u_realfla).float(), requires_grad=True)
u_real_numpy = u_real_torch.data.cpu().numpy()

mse_torch = mse_cost_function1(u_real_torch, unn_torch)
mse_numpy = mse_torch.data.cpu().numpy()

error = u_real_numpy - unn_numpy
error_mean = np.mean(np.abs(error))

unn_matrix = unn_numpy.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)
u_real_matrix = u_real_numpy.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)
error_matrix = error.reshape(test_M1 + 1, test_M2 + 1, test_N + 1)

print("error max:", (np.abs(error)).max())
print("error mean:", error_mean)
error_L2 = np.linalg.norm(error, ord=2) / np.linalg.norm(u_real_numpy, ord=2)
print("error L2:", error_L2)
print("error mse:", mse_numpy)

np.savetxt('2D_error.txt', (error))
np.savetxt('2D_unn.txt', (unn_numpy))
np.savetxt('2D_ureal.txt', (u_real_numpy))

u_real_t0 = u_real_matrix[:, :, 10]
unn_t0 = unn_matrix[:, :, 10]
error_t0 = error_matrix[:, :, 10]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x1_plot, x2_plot, unn_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.savefig('udata_fpde_2d_t01.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, error_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('error')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='z')
plt.savefig('errordata_fpde_2d_t01.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, u_real_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率

plt.savefig('u_real_fpde_2d_t01.png')
plt.close(fig)

u_real_t0 = u_real_matrix[:, :, 50]
unn_t0 = unn_matrix[:, :, 50]
error_t0 = error_matrix[:, :, 50]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x1_plot, x2_plot, unn_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.savefig('udata_fpde_2d_t05.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, error_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('error')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='z')
plt.savefig('errordata_fpde_2d_t05.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, u_real_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率

plt.savefig('u_real_fpde_2d_t05.png')
plt.close(fig)

u_real_t0 = u_real_matrix[:, :, 90]
unn_t0 = unn_matrix[:, :, 90]
error_t0 = error_matrix[:, :, 90]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.plot_surface(x1_plot, x2_plot, unn_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.savefig('udata_fpde_2d_t09.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, error_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('error')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率
plt.ticklabel_format(style='sci', scilimits=(0, 0), axis='z')
plt.savefig('errordata_fpde_2d_t09.png')
plt.close(fig)

fig = plt.figure()
bx = fig.add_subplot(projection='3d')
bx.plot_surface(x1_plot, x2_plot, u_real_t0, cmap=cm.RdYlBu_r, edgecolor='blue', linewidth=0.0003, antialiased=True)
bx.set_xlabel('x')
bx.set_ylabel('y')
bx.set_zlabel('u')
plt.rcParams['savefig.dpi'] = 500  # 图片像素
plt.rcParams['figure.dpi'] = 500  # 分辨率

plt.savefig('u_real_fpde_2d_t09.png')
plt.close(fig)