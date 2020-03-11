import torch
import torchvision
from models import ResNet18

# loading the model
path = "pretrained/pretrained88.pth"
m = ResNet18()
m.load_state_dict(torch.load(path))


# load test set
transform = torchvision.transforms.ToTensor()
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                         shuffle=False, num_workers=0)


# evaluate clean accuracy (=90.84)
# since images are unnormalized (between 0 and 1), we want the model to
# perform the normalization internally
m.set_normalize(True)
m.eval()
correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = m(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('Accuracy of the network on clean test set: %.3f %%' % (
    100 * correct / total))

