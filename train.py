import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import aim
from load_dataset import imshow

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 18, 5)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(18, 48, 3)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(48 * 4 * 4, 512)

        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 62)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x)
        return x

def train_torch(device, modelPath, trainloader, valloader, epochs, lr, lr_cd, lr_f, lr_p, momentum, epochsPerSave):
    aim_run = aim.Run(repo='.')
    aim_run.name = modelPath
    net = Net().to(device)
    print(net)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode='min', cooldown=lr_cd, factor=lr_f, patience=lr_p)
    for epoch in range(epochs):
        running_loss = 0.0
        steps = 0.0
        for data in trainloader:
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
            steps += 1.0

        val_inputs, val_labels = next(iter(valloader))
        val_outputs = net(val_inputs)
        val_loss = criterion(val_outputs, val_labels)
        scheduler.step(running_loss/steps)

        aim_run.track(val_loss.item(), 'val_loss', epoch=epoch)
        aim_run.track(running_loss/steps, 'loss', epoch=epoch)
        print('[%d]'%(epoch+1),'\tloss\t', running_loss/steps, 'val_loss\t', val_loss.item(), 'lr\t', scheduler.get_last_lr())

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

def inference_torch(image, net):
    image = torch.Tensor(np.expand_dims(np.transpose(image, (2,0,1)),axis=0))   
    outputs = net(image)
    probability, predicted = torch.max(outputs, 1)
    return predicted.item(), probability.item()
