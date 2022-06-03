import torch
from torch import nn
from torch import optim
from model import ConvNet
from datahandler import trainloader, testloader


model = ConvNet()
device = 'cuda' if torch.cuda.is_available() else 'cpu'
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.NLLLoss()

train_losses=[]
test_losses=[]
acc_list = []

epochs=15


for epoch in range(epochs):
    train_batch_loss = []
    for images, labels in iter(trainloader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        output = model.forward(images)
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()
        train_batch_loss.append(loss.item())
    
    mean_train_batch_loss = sum(train_batch_loss)/len(train_batch_loss)
    train_losses.append(mean_train_batch_loss)

    model.eval()
    with torch.no_grad():
        batch_test_loss=[]
        for t_images, t_labels in iter(testloader):
            t_images, t_labels= t_images.to(device), t_labels.to(device)
            logprob = model(t_images)
            probability = torch.exp(logprob)
            pred = probability.argmax(dim=1)
            test_loss = criterion(logprob, t_labels)
            acc = (pred == t_labels).sum() / len(t_labels) * 100
            acc_list.append(acc)
        model.train()
    print(f'Epoch: {epoch+1}/{epochs} | Acccuracy: {sum(acc_list)/len(acc_list)} | Train-Loss: {train_losses[epoch]} | Done!')