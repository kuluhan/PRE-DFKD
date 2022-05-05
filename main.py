import os
import random
import argparse
import numpy as np
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from networks.generator import Generator, Encoder, MemoryGenerator
from utils.data_utils import load_val_dataset
from utils.model_utils import init_model, eval_model
from utils.constants import ROOT_DIR, num_channels
from distillation import distill_knowledge, train_novel_generator, train_memory_generator

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='cifar100', choices=['MNIST','cifar10','cifar100','SVHN','FASHION','tiny-imagenet'])
parser.add_argument('--student', type=str, default='resnet18', choices=['lenet','mobilenet','resnet34','resnet18','squeezenet'])
parser.add_argument('--data', type=str, default=os.path.join(ROOT_DIR,'cache/data/'))
parser.add_argument('--teacher_dir', type=str, default=os.path.join(ROOT_DIR,'cache/models/'))
parser.add_argument('--teacher_name', type=str, default='cifar100_teacher_resnet34')
parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs of training')
parser.add_argument('--gen_iter', type=int, default=1, help='# of generator training loop iterations')
parser.add_argument('--kd_iter', type=int, default=10, help='# of kd iterations')
parser.add_argument('--batch_size', type=int, default=1024, help='size of the batches')
parser.add_argument('--lr_G', type=float, default=0.02, help='learning rate')
parser.add_argument('--lr_S', type=float, default=0.1, help='learning rate')
parser.add_argument('--latent_dim', type=int, default=1000, help='dimensionality of the latent space')
parser.add_argument('--img_size', type=int, default=32, help='size of each image dimension')
parser.add_argument('--oh', type=float, default=1, help='one hot loss')
parser.add_argument('--ie', type=float, default=5, help='information entropy loss')
parser.add_argument('--a', type=float, default=0.1, help='activation loss')
parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR,'cache/models/'))
opt = parser.parse_args()

# set manual random seed
seed = 1
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Initialize generators 
novel_generator = Generator(opt, num_channels).cuda()
mem_generator = MemoryGenerator(opt, num_channels).cuda()
encoder = Encoder(opt, num_channels).cuda()

novel_generator = nn.DataParallel(novel_generator)
mem_generator = nn.DataParallel(mem_generator)
encoder = nn.DataParallel(encoder)

optimizer_G = torch.optim.Adam( novel_generator.parameters(), lr=opt.lr_G )
optimizer_G2 = torch.optim.Adam( mem_generator.parameters(), lr=opt.lr_G / 40 )
optimizer_E = torch.optim.Adam( encoder.parameters(), lr=opt.lr_G / 40 )
    
# Initialize student
student = init_model(opt)
student = nn.DataParallel(student)

data_test, data_test_loader = load_val_dataset(opt)

if opt.dataset == 'MNIST':
    optimizer_S = torch.optim.Adam(student.parameters(), lr=opt.lr_S)  
else:
    optimizer_S = torch.optim.SGD(student.parameters(), lr=opt.lr_S, momentum=0.9, weight_decay=5e-4)   
scheduler_S = torch.optim.lr_scheduler.CosineAnnealingLR( optimizer_S, T_max=400)

# Load teacher
teacher = torch.load(opt.teacher_dir + opt.teacher_name).cuda()
teacher = nn.DataParallel(teacher)
teacher.eval()
criterion = torch.nn.CrossEntropyLoss().cuda()

# -------------
#  Distillation
# -------------
print('DISTILLATION STARTED')
accr = 0
opt.num_novel_samples = opt.batch_size
opt.num_mem_samples = (opt.batch_size // 8)
opt.mem_gen_upd_period = 1
opt.student_rehearse_period = 1

start_time = time.time()

for epoch in range(opt.n_epochs):
    student.train()
    
    for i in range(36):
        # Train novel generator
        for k in range(opt.gen_iter):
            novel_gen_losses = train_novel_generator(opt, novel_generator, optimizer_G, teacher, student)

        # Distill Knowledge
        for dist_step in range(opt.kd_iter):
            # Generate new synthetic samples
            z = Variable(torch.randn(opt.num_novel_samples, opt.latent_dim), requires_grad=False).cuda()
            gen_imgs = novel_generator(z)

            loss_kd = distill_knowledge(opt, epoch, student, teacher, mem_generator, optimizer_S, gen_imgs)

            # Memory Generator Training
            if (epoch % opt.mem_gen_upd_period == 0) & (dist_step < 4):
                mem_gen_loss = train_memory_generator(opt, epoch, i, mem_generator, encoder, optimizer_G2, optimizer_E, teacher, gen_imgs)

        if i == 1:
            print("[Epoch %d/%d] [loss_G: %f] [loss_G2: %f] [loss_S: %f]" % (epoch, opt.n_epochs, mem_gen_loss.item(), novel_gen_losses[0].item(), loss_kd.item()))
            print('Time elapsed: %.2f seconds' % (time.time() - start_time))

    scheduler_S.step()
    accr = eval_model(student, data_test, data_test_loader)
