import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import matplotlib.pyplot as plt
from util import mysampler

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 设置超参数
batch_size = 32
learning_rate = 1e-4
num_epochs = 50

transform = transforms.Compose(
    [
        transforms.Resize([224, 224]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),  # 在（-10， 10）,  # 范围内旋转
        transforms.ColorJitter(0.05, 0.05, 0.05, 0.05),  # HSV以及对比度变化
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
root = "./data/"
datasets = {
    name: torchvision.datasets.ImageFolder(root + name, transform=transform)
    for name in ["train_set", "val_set", "test_set"]
}

train_set = DataLoader(datasets["train_set"], sampler=mysampler.ImbalancedDatasetSampler(datasets["train_set"]), batch_size=batch_size)
val_set = DataLoader(datasets["val_set"], sampler=mysampler.ImbalancedDatasetSampler(datasets["val_set"]), batch_size=batch_size)
test_set = DataLoader(datasets["test_set"], 1)


class mymobilenet(nn.Module):
    def __init__(self, num_classes=2):
        super(mymobilenet, self).__init__()
        self.mobilenet = torchvision.models.mobilenet_v2(pretrained=True)
        self.dropout = nn.Dropout(0.2)
        self.layer = nn.Linear(1000, num_classes)

    def forward(self, x):
        out = self.mobilenet(x)
        out = self.dropout(out)
        out = self.layer(out)
        return out


model = mymobilenet(num_classes=2).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=0.01)


def train():
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    for epoch in range(num_epochs):
        model.train()
        train_running_loss = 0
        train_corrects = 0
        for images, labels in train_set:
            images = images.to(device)
            labels = labels.to(device)
            output = model(images)
            _, pred = torch.max(output, 1)
            train_corrects += torch.sum(pred == labels.data).item()
            loss = criterion(output, labels)
            train_running_loss += loss.item() * images.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        train_running_loss /= len(datasets["train_set"])
        train_loss.append(train_running_loss)
        train_running_acc = train_corrects / len(datasets["train_set"])
        train_acc.append(train_running_acc)

        model.eval()
        val_running_loss = 0
        val_corrects = 0
        for images, labels in val_set:
            images = images.to(device)
            labels = labels.to(device)
            with torch.no_grad():
                output = model(images)
                loss = criterion(output, labels)
                output = model(images)
                _, pred = torch.max(output, 1)
                val_corrects += torch.sum(pred == labels.data).item()
                loss = criterion(output, labels)
                val_running_loss += loss.item() * images.size(0)
        val_running_loss /= len(datasets["val_set"])
        val_loss.append(val_running_loss)
        val_running_acc = val_corrects / len(datasets["val_set"])
        val_acc.append(val_running_acc)
        print(
            "epoch[{}] train_loss:{:.4f} train_acc:{:.2f}% val_loss:{:.4f} val_acc:{:.2f}%".format(
                epoch + 1,
                train_running_loss,
                train_running_acc * 100,
                val_running_loss,
                val_running_acc * 100,
            )
        )

    print("finish training!")

    plt.subplot(2, 1, 1)
    plt.title("train and validate loss")
    plt.plot(train_loss, "o", label="train_loss")
    plt.plot(val_loss, "o", label="val_loss")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")

    plt.subplot(2, 1, 2)
    plt.title("train and validate acc")
    plt.plot(train_acc, "o-", label="train_acc")
    plt.plot(val_acc, "o-", label="val_acc")
    plt.xlabel("epoch")
    plt.legend(loc="lower right")
    plt.gcf().set_size_inches(15, 12)
    plt.savefig("MobileNetV2_bs32_lr1e-4_epoch50_reg1e-2 training curve")
    # plt.show()


def test():
    running_loss = 0
    corrects = 0
    model.eval()
    for image, label in test_set:
        images = image.to(device)
        labels = label.to(device)
        with torch.no_grad():
            output = model(images)
            loss = criterion(output, labels)
            _, pred = torch.max(output, 1)
            corrects += torch.sum(pred == labels.data).item()
            loss = criterion(output, labels)
            running_loss += loss.item() * images.size(0)
    print(
        "TEST loss:{:.4f} acc:{:.2f}%".format(
            running_loss / len(datasets["test_set"]), corrects / len(datasets["test_set"]) * 100
        )
    )
    torch.save(model.state_dict(), "mobilenetv2_bs32_lr1e-4_epoch50_reg1e-2.pickle")


if __name__ == "__main__":
    train()
    test()
