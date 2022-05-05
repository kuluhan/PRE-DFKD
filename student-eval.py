import os
import argparse

import torch
import torch.nn.functional as F

from utils.constants import ROOT_DIR
from utils.data_utils import load_val_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='MNIST', choices=['MNIST','cifar10','cifar100','SVHN','FASHION', 'tiny-imagenet'])
parser.add_argument('--data', type=str, default=os.path.join(ROOT_DIR,'cache/data/'))
parser.add_argument('--student_dir', type=str, default=os.path.join(ROOT_DIR,'cache/models/'))
parser.add_argument('--student_name', type=str)
parser.add_argument('--batch_size', type=int, default=512, help='size of the batches')
opt = parser.parse_args()

student = torch.load(opt.student_dir + opt.student_name).cuda()
student.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

data_test, data_test_loader = load_val_dataset(opt)

avg_loss = 0
total_correct = 0
with torch.no_grad():
    for i, (images, labels) in enumerate(data_test_loader):
        images = images.cuda()
        labels = labels.cuda()
        student.eval()
        output = student(images)
        softmax_o_S = F.softmax(output, dim=1)
        avg_loss += criterion(output, labels).sum()
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.data.view_as(pred)).sum()

avg_loss /= len(data_test)
print('Test Avg. Loss: %f, Accuracy: %f' % (avg_loss.data.item(), float(total_correct) / len(data_test)))
