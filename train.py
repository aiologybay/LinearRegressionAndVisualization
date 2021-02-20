import matplotlib.pyplot as plt
import torch
import torch.nn as nn

x = torch.randn(1024, 1)
y = 2 * x + 3

print(x)
print(y)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        x = self.linear(x)
        return x


net = Net()
print(net)

optimizer = torch.optim.SGD(net.parameters(), lr=0.003)
criterion = torch.nn.MSELoss()

loss_his = []
for epoch in range(100000):
    y_pred = net(x)
    loss = criterion(y_pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    loss_his.append(loss.data.numpy())
    if epoch % 100 == 0:
        plt.cla()
        plt.subplot(2, 1, 1)
        plt.title('Fitting Line')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.plot(x, y, 'bo', label='Train Data', ms=2)
        plt.plot(x, y_pred.data.numpy(), 'r', label='Fitting')
        plt.text(1, 0, 'w={:.2f}\nb={:.2f}'.format(net.linear.weight.data.numpy().item(),
                                                   net.linear.bias.data.numpy().item()),
                 fontdict={'size': 10, 'color': 'red'})
        plt.legend(loc='upper left')
        # plt.grid()
        # plt.axis('off')
        plt.subplot(2, 1, 2)
        plt.title('Loss Curve')
        plt.xlabel('iteration')
        plt.ylabel('loss')
        plt.plot(loss_his, 'black', label='SGD', linewidth=1)
        plt.text(0, 4, 'Loss={:.6f}'.format(loss.data), fontdict={'size': 10, 'color': 'red'})
        plt.legend(loc='upper right')
        # plt.grid()
        # plt.axis('off')
        plt.pause(0.5)
        plt.clf()
