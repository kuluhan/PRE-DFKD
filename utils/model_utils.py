import torch

from utils.constants import num_classes, num_channels
from networks import lenet, resnet, squeezenet, mobilenet

def eval_model(net, data_test, data_test_loader):
    total_correct = 0
    avg_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images = images.cuda()
            labels = labels.cuda()
            net.eval()
            output = net(images)
            avg_loss += torch.nn.CrossEntropyLoss()(output, labels).sum()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()

    avg_loss /= len(data_test)
    print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
    accr = round(float(total_correct) / len(data_test), 4)
    return accr

def init_model(opt):
    if opt.student == 'lenet':
        net = lenet.LeNet5Half(num_classes=num_classes[opt.dataset]).cuda()
    elif opt.student == 'mobilenet':
        net = mobilenet.MobileNetV2(num_classes=num_classes[opt.dataset], in_channels=num_channels[opt.dataset]).cuda()
    elif opt.student == 'resnet34':
        net = resnet.ResNet34(num_classes=num_classes[opt.dataset], in_channels=num_channels[opt.dataset]).cuda()
    elif opt.student == 'resnet18':
        net = resnet.ResNet18(num_classes=num_classes[opt.dataset], in_channels=num_channels[opt.dataset]).cuda()
    elif opt.student == 'squeezenet':
        net = squeezenet.SqueezeNet(num_classes=num_classes[opt.dataset], in_channels=num_channels[opt.dataset]).cuda()
    return net