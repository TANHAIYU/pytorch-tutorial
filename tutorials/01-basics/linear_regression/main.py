import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


# Hyper-parameters
input_size = 1
output_size = 1
num_epochs = 60
learning_rate = 0.001

# Toy dataset 导入对应得包
x_train = np.array([[3.3], [4.4], [5.5], [6.71], [6.93], [4.168], 
                    [9.779], [6.182], [7.59], [2.167], [7.042], 
                    [10.791], [5.313], [7.997], [3.1]], dtype=np.float32)

y_train = np.array([[1.7], [2.76], [2.09], [3.19], [1.694], [1.573], 
                    [3.366], [2.596], [2.53], [1.221], [2.827], 
                    [3.465], [1.65], [2.904], [1.3]], dtype=np.float32)

# Linear regression model
model = nn.Linear(input_size, output_size)

# Loss and optimizer  定义loss和优化函数
criterion = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)  

# Train the model
for epoch in range(num_epochs):
    # Convert numpy arrays to torch tensors
    # 之后我们将数据库中的点都输入进来，并且使用torch.from_numpy（）方法将numpy中的ndarray转化成pytorch中的tensor。
    inputs = torch.from_numpy(x_train)
    targets = torch.from_numpy(y_train)
# 首先设epoch总数是1000，之后对每个epoch进行遍历。
# 之后我们开始前向传播求输出结果和整个的损失loss，之后我们使用反向传播梯度下降来更新参数。
# 每当我们的epoch数目可以整除20的时候，我们将这个epoch的loss输出，以便我们可以了解到loss的变化情况。

    # Forward pass
    outputs = model(inputs)
    loss = criterion(outputs, targets)
    
    # Backward and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch+1) % 5 == 0:
        print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))

# Plot the graph
predicted = model(torch.from_numpy(x_train)).detach().numpy()
plt.plot(x_train, y_train, 'ro', label='Original data')
plt.plot(x_train, predicted, label='Fitted line')
plt.legend() # 显示图例
plt.show()

# Save the model checkpoint 保存模型
torch.save(model.state_dict(), 'model.ckpt')