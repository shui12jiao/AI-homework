import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

BATCH_SIZE = 128
EPOCHS = 10
DEVICE = torch.device("cpu")

# if torch.cuda.is_available():
#     DEVICE = torch.device("cuda")
#     cudnn.benchmark = True

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)
# 测试集

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1037,), (0.3081,))]
        ),
    ),
    batch_size=BATCH_SIZE,
    shuffle=True,
)


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, 5)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(20 * 10 * 10, 500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        in_size = x.size(0)
        out = self.conv1(x)  # 1* 10 * 24 *24
        out = F.relu(out)
        out = F.max_pool2d(out, 2, 2)  # 1* 10 * 12 * 12
        out = self.conv2(out)  # 1* 20 * 10 * 10
        out = self.conv2_drop(out)
        out = F.relu(out)
        out = out.view(in_size, -1)  # 1 * 2000
        out = self.fc1(out)  # 1 * 500
        out = F.relu(out)
        out = self.fc2(out)  # 1 * 10
        out = F.log_softmax(out, dim=1)
        return out


model = ConvNet().to(DEVICE)
optimizer = optim.Adam(model.parameters())


def train(model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(input=output, target=target).to(DEVICE)
        loss.backward()
        optimizer.step()
        if (batch_idx + 1) % 30 == 0:
            print(
                "Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                    epoch,
                    batch_idx * len(data),
                    len(train_loader.dataset),
                    100.0 * batch_idx / len(train_loader),
                    loss.item(),
                )
            )


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction="sum")  # 将一批的损失相加
            pred = output.max(1, keepdim=True)[1]  # 找到概率最大的下标
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%) \n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )


for epoch in range(1, EPOCHS + 1):
    train(model, DEVICE, train_loader, optimizer, epoch)
    test(model, DEVICE, test_loader)
