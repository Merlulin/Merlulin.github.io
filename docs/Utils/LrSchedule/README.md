# 学习率调度器（lr_schedule）

## ReducelROnPlateau

> ReduceLROnPlateau是PyTorch中的一个动态学习率算法，它根据训练集上的loss值来自动调整学习率。<br> 该算法会在训练的过程中监控损失函数的值，并且在损失降低的速度变慢的时候，自动的减少学习率。当损失函数连续patient轮迭代都没有下降时，学习率则会减少一个因子factor

**主要参数：**
1. Optimizer： 被包装的优化器
2. mode： （'min' || 'max'）在min模式下，当检测到被监控的参数不再减少时降低学习率；在max模式下，当检测到没监控的参数不再增加时降低学习率。默认值：'min'
3. factor： 学习率降低的因子 -> new_lr = lr * factor
4. patience: 当监控的参数在patience个数的epoch内没有改善，则再次之后降低学习率。—> 如果 Patience = 2，那么我们将忽略前 2 个没有改善的 epoch，并且只有在第 3 个 epoch 之后损失仍然没有改善时才会降低学习率。默认值：10。

**示例代码：**
```python
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5)
for epoch in range(num_epochs):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    train_loss /= len(train_loader)
    lr_scheduler.step(train_loss)
```