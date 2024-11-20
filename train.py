import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from load_dataset import imshow

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 3)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 4)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train_torch(modelPath, trainloader, epochs=15, lr=0.05, momentum=0.5, epochsPerSave=5, elsPerStat=50):
    net = Net()
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % elsPerStat == (elsPerStat-1):    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / elsPerStat:.3f}')
                running_loss = 0.0
        if epoch % epochsPerSave == (epochsPerSave-1):
            torch.save(net.state_dict(), modelPath.replace('.pth', '_%se.pth'%epoch))
    print('Finished Training')
    torch.save(net.state_dict(), modelPath)

    metadata = open(modelPath+'.meta', 'w')
    metadata.write(net)
    metadata.close()

def test_torch(modelPath, testloader, classes, num=4):
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    net = Net()
    net.load_state_dict(torch.load(modelPath, weights_only=True))
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    accuracy = 0.0
    for i in range(len(labels)):
        if labels[i]==predicted[i]:
            accuracy += 100.0/len(labels)
    print('GroundTruth', 'Predicted', '(accuracy - %f%%)'%accuracy)
    print('\n'.join(f'{classes[labels[j]]:5s} {classes[predicted[j]]:5s}' for j in range(num)))
    imshow(torchvision.utils.make_grid(images))
