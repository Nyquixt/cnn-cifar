import torch

def calculate_acc(dataloader, net):
    with torch.no_grad():
        correct = 0
        total = 0
        for data in dataloader:
            images, labels = data
            images = images.cuda()
            labels = labels.cuda()
            
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    return (correct / total) * 100