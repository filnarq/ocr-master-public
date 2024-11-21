import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
from load_dataset import imshow

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(18, 48, 3)
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

def train_torch(device, modelPath, trainloader, epochs, lr, lr_gamma, lr_gamma_steps, momentum, epochsPerSave, elsPerStat):
    net = Net().to(device)
    print(net)
    minLoss = 1000000.0
    incrLoss = 0
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=lr_gamma_steps, gamma=lr_gamma)
    for epoch in range(epochs):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            # check batch stats
            if i % elsPerStat == (elsPerStat-1):

                # Adjust learning rate
                if (running_loss/elsPerStat) > minLoss:
                    incrLoss += 1
                else:
                    minLoss = running_loss/elsPerStat
                    incrLoss = 0
                if incrLoss > 3:
                    scheduler.step()
                    minLoss = running_loss/elsPerStat
                    incrLoss = 0
                    print('lr', scheduler.get_last_lr())
                running_loss = 0.0

                # Print
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / elsPerStat:.3f} | incrLoss {incrLoss} minLoss {minLoss}')
        
        # Save checkpoint
        if epoch % epochsPerSave == (epochsPerSave-1):
            torch.save(net.state_dict(), modelPath.replace('.pth', '_%se.pth'%(epoch+1)))
    
    # Save model
    print('Finished Training')
    torch.save(net.state_dict(), modelPath)

def test_torch(device, modelPath, testloader, classes, batchSize):
    # Get first batch
    dataiter = iter(testloader)
    images, labels = next(dataiter)

    # Load net and get prediction
    net = Net()
    net.load_state_dict(torch.load(modelPath, weights_only=True, map_location=device))
    net.eval()
    outputs = net(images)
    _, predicted = torch.max(outputs, 1)

    # Calculate accuracy
    accuracy = 0.0
    for i in range(len(labels)):
        if labels[i]==predicted[i]:
            accuracy += 100.0/len(labels)
    print('GroundTruth', 'Predicted', '(accuracy - %f%%)'%accuracy)
    print('\n'.join(f'{classes[labels[j]]:5s} {classes[predicted[j]]:5s}' for j in range(batchSize)))
    imshow(torchvision.utils.make_grid(images))
