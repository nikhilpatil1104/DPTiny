import numpy as np

from dptiny import (
    Variable,
    is_available,
    is_gpu,
    no_grad,
    softmax_cross_entropy,
    test_mode,
    to_cpu,
    to_gpu,
    use_gpu,
    xp,
)
from dptiny.data import DataLoader
from dptiny.data.fashion_mnist import CLASSES, get_fashion_mnist
from dptiny.nn import (
    Conv2d,
    Dropout,
    Flatten,
    Linear,
    MaxPool2d,
    ReLU,
    Sequential,
)
from dptiny.optim import Adam

if is_available():
    use_gpu()
    print("GPU enabled for training.")
else:
    print("GPU not available; training on CPU.")

xp.random.seed(0)

print("Loading Fashion-MNIST dataset...")
X_train, X_test, y_train, y_test = get_fashion_mnist(flatten=False)

mean = X_train.mean()
std = X_train.std()
X_train = ((X_train - mean) / std).astype(np.float32)
X_test = ((X_test - mean) / std).astype(np.float32)

if is_gpu():
    X_train = to_gpu(X_train)
    X_test = to_gpu(X_test)
    y_train = to_gpu(y_train)
    y_test = to_gpu(y_test)

model = Sequential(
    Conv2d(1, 16, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Conv2d(16, 32, 3, pad=1),
    ReLU(),
    MaxPool2d(2),
    Flatten(),
    Linear(32 * 7 * 7, 128),
    ReLU(),
    Dropout(0.3),
    Linear(128, 10),
)
if is_gpu():
    model.to_gpu()

batch_size = 64
max_epoch = 10
data_loader = DataLoader((X_train, y_train), batch_size)
test_loader = DataLoader((X_test, y_test), batch_size, shuffle=False)

optimizer = Adam(model, lr=0.001)

for epoch in range(max_epoch):
    sum_loss = 0.0
    sum_correct = 0.0
    count = 0
    model.train()
    for x, t in data_loader:
        x = Variable(x)
        y = model(x)
        loss = softmax_cross_entropy(y, t)

        model.cleargrads()
        loss.backward()
        optimizer.update()

        sum_loss += float(loss.data) * len(t)
        sum_correct += float((y.data.argmax(axis=1) == t).sum())
        count += len(t)

    test_correct = 0.0
    test_count = 0
    with test_mode(), no_grad():
        for x, t in test_loader:
            y = model(Variable(x))
            pred = y.data.argmax(axis=1)
            test_correct += float((pred == t).sum())
            test_count += len(t)
    print(
        f"epoch: {epoch + 1}, train loss: {sum_loss / count:.4f}, "
        f"train acc: {sum_correct / count:.4f}, "
        f"test acc: {test_correct / test_count:.4f}"
    )

confusion = np.zeros((10, 10), dtype=np.int64)
with test_mode(), no_grad():
    for x, t in test_loader:
        y = model(Variable(x))
        pred = to_cpu(y.data.argmax(axis=1))
        np.add.at(confusion, (to_cpu(t), pred), 1)

print("\nConfusion matrix (rows: true, columns: predicted):")
print(confusion)

print("\nPer-class accuracy:")
for i, name in enumerate(CLASSES):
    print(f"{name}: {confusion[i, i] / confusion[i].sum():.4f}")

best_pair = None
best_count = -1
for i in range(10):
    for j in range(i + 1, 10):
        n = confusion[i, j] + confusion[j, i]
        if n > best_count:
            best_pair, best_count = (i, j), n

print(
    f"\nMost confused pair: {CLASSES[best_pair[0]]} and {CLASSES[best_pair[1]]} "
    f"({best_count} mistakes)"
)
