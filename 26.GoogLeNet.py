import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm


class Inception(nn.Module):
    """
    Inception网络模块，用于深度学习中的多尺度特征提取。

    该模块通过四个不同的分支对输入进行并行处理，每个分支负责不同尺度的特征提取，
    最后将各个分支的输出合并，形成最终的输出。这种设计使得网络能够同时捕捉到不同尺度
    和类型的信息。

    参数:
    - in_channels: 输入通道数。
    - ch1x1: 第一个分支中1x1卷积层的输出通道数。
    - ch3x3red: 第二个分支中用于降维的1x1卷积层的输出通道数。
    - ch3x3: 第二个分支中3x3卷积层的输出通道数。
    - ch5x5red: 第三个分支中用于降维的1x1卷积层的输出通道数。
    - ch5x5: 第三个分支中5x5卷积层的输出通道数。
    - pool_proj: 第四个分支中用于池化层后降维的1x1卷积层的输出通道数。
    """

    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(Inception, self).__init__()

        # 第一个分支：1x1卷积层，用于捕捉细节特征
        self.branch1 = nn.Conv2d(in_channels, ch1x1, kernel_size=1)

        # 第二个分支：先通过1x1卷积层降维，再通过3x3卷积层捕捉边缘等中等尺度特征
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )

        # 第三个分支：先通过1x1卷积层降维，再通过5x5卷积层捕捉更大尺度的特征
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )

        # 第四个分支：通过最大池化层提取低频特征，再通过1x1卷积层降维
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        """
        前向传播函数，将输入通过四个分支进行处理，并将结果合并。

        参数:
        - x: 输入张量。

        返回:
        - 输出张量：四个分支输出的合并。
        """
        # 通过四个分支分别进行前向传播
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)

        # 将四个分支的输出在通道维度上合并
        return torch.cat([branch1, branch2, branch3, branch4], dim=1)
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.conv = nn.Conv2d(in_channels, 128, kernel_size=1)
        self.fc1 = nn.Linear(2048, 1024)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10, aux_logits=True):
        super(GoogLeNet, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.inception3a = Inception(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = Inception(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = Inception(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = Inception(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = Inception(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = Inception(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = Inception(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception5a = Inception(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = Inception(832, 384, 192, 384, 48, 128, 128)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024, num_classes)

        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)
        x = self.inception4a(x)
        aux1 = self.aux1(x) if self.aux_logits else None
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        aux2 = self.aux2(x) if self.aux_logits else None
        x = self.inception4e(x)
        x = self.maxpool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        if self.aux_logits and self.training:
            return x, aux1, aux2
        return x
def load_data_fashion_mnist(batch_size, resize=None):
    trans = [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    if resize:
        trans.insert(0, transforms.Resize(resize))
    transform = transforms.Compose(trans)
    mnist_train = datasets.FashionMNIST(root='../data', train=True, transform=transform, download=True)
    mnist_test = datasets.FashionMNIST(root='../data', train=False, transform=transform, download=True)
    return DataLoader(mnist_train, batch_size=batch_size, shuffle=True), DataLoader(mnist_test, batch_size=batch_size, shuffle=False)
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        # 仅使用主输出计算损失
        outputs_main = outputs[0] if isinstance(outputs, tuple) else outputs
        loss = criterion(outputs_main, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        # 假设 outputs 是 (output_tensor, other_info)
        output_tensor = outputs[0]  # 取出第一个元素作为真正的输出
        _, predicted = output_tensor.max(1)

        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {avg_loss:.4f} Accuracy: {acc:.2f}%')
    return avg_loss, acc
def test(model, device, test_loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / total
    acc = 100. * correct / total
    print(f'Test Loss: {avg_loss:.4f}, Accuracy: {acc:.2f}%')
    return avg_loss, acc
def main():
    batch_size = 128
    epochs = 20
    lr = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 调整输入尺寸为 224x224，因为 GoogLeNet 原始设计是用于 ImageNet 的
    train_loader, test_loader = load_data_fashion_mnist(batch_size, resize=224)

    model = GoogLeNet(num_classes=10, aux_logits=True).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        test(model, device, test_loader, criterion)

if __name__ == '__main__':
    main()
