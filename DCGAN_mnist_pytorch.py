from __future__ import print_function

import argparse
import numpy as np
import os
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import torchvision.utils as vutils
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--outf', default='./output', help='folder to put model generate image')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

random.seed(args.seed)
torch.manual_seed(args.seed)

try:
    os.makedirs(args.outf)
except OSError:
    pass

print('loading data!')
trainset_unlabeled = pickle.load(open("data/train_unlabeled.p", "rb"))
trainset_labeled = pickle.load(open("data/train_labeled.p", "rb"))
validset = pickle.load(open("data/validation.p", "rb"))

unsupervised_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64, shuffle=True, **kwargs)
supervised_loader = torch.utils.data.DataLoader(trainset_labeled, batch_size=64, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset, batch_size=64, shuffle=True)


# Randomly intialize Conv and BatchNorm layer to break symmetry
def weights_init(m):
  classname = m.__class__.__name__
  if classname.find('Conv') != -1:
    m.weight.data.normal_(0.0, 0.02)
  elif classname.find('BatchNorm') != -1:
    m.weight.data.normal_(1.0, 0.02)
    m.bias.data.fill_(0)


class DiscriminatorNet(nn.Module):
  def __init__(self):
    super(DiscriminatorNet, self).__init__()
    self.features = nn.Sequential(
      nn.Conv2d(1, 16, kernel_size=4, stride=2),
      nn.BatchNorm2d(16),
      nn.LeakyReLU(0.2),
      nn.Conv2d(16, 32, kernel_size=5, stride=2),
      nn.BatchNorm2d(32),
      nn.LeakyReLU(0.2)
    )
    self.classifier = nn.Sequential(
      nn.Linear(800, 1),
      nn.Sigmoid()
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(-1, 800)
    x = self.classifier(x)
    return x


class GeneratorNet(nn.Module):
  def __init__(self):
    super(GeneratorNet, self).__init__()
    self.noise = nn.Sequential(
      nn.Linear(20, 800),
      nn.ReLU(inplace=True)
    )
    self.main = nn.Sequential(
      nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2),
      nn.BatchNorm2d(16),
      nn.ReLU(inplace=True),
      nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2),
      nn.Tanh()
    )

  def forward(self, z):
    x = self.noise(z)
    x = x.view(-1, 32, 5, 5)
    x = self.main(x)
    return x


G = GeneratorNet()
D = DiscriminatorNet()
G.apply(weights_init)
D.apply(weights_init)
D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
G_optimizer = optim.Adam(G.parameters(), lr=args.lr)
fixed_noise = Variable(torch.randn(1, 20))

for epoch in range(1, args.epochs + 1):
  for i, (x, _) in enumerate(unsupervised_loader):
    ############################
    # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
    ###########################
    D_optimizer.zero_grad()
    x = Variable(x)
    x_output = D(x)
    z = Variable(torch.randn(x.size(0), 20))
    gz = G(z)
    gz_output = D(gz)
    d_loss = -(torch.mean(torch.log(x_output) + torch.log(1 - gz_output)))
    d_loss.backward()
    D_optimizer.step()

    ############################
    # (2) Update G network: maximize log(D(G(z)))
    ###########################
    G_optimizer.zero_grad()
    z = Variable(torch.randn(x.size(0), 20))
    gz = G(z)
    gz_output = D(gz)
    g_loss = -torch.mean(torch.log(gz_output))
    g_loss.backward()
    G_optimizer.step()

    if i % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(x), len(unsupervised_loader.dataset),
            100. * i / len(unsupervised_loader), d_loss.data[0]))
    if i % 100 == 0:
      fake = G(fixed_noise)
      vutils.save_image(fake.data,
        '{}/fake_samples_epoch_{}.png'.format(args.outf, epoch))


# Save ckpt at last epoch
torch.save(G.state_dict(), '{}/G_epoch_{}.pth'.format(args.outf, epoch))
torch.save(D.state_dict(), '{}/D_epoch_{}.pth'.format(args.outf, epoch))

D.classifier = torch.nn.Sequential(
  nn.Linear(800, 10),
  nn.ReLU(inplace=True),
  nn.LogSoftmax()
)

optimizer = optim.SGD(D.parameters(), lr=args.lr, momentum=args.momentum)

for epoch in range(1, args.epochs + 1):
  for i, (data, target) in enumerate(supervised_loader):
    data, target = Variable(data), Variable(target)
    optimizer.zero_grad()
    output = D(data)
    loss = F.nll_loss(output, target)
    loss.backward()
    optimizer.step()
    if i % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, i * len(data), len(supervised_loader.dataset),
        100. * i / len(supervised_loader), loss.data[0]))

D.eval()
test_loss = 0
correct = 0
for data, target in valid_loader:
  data, target = Variable(data, volatile=True), Variable(target)
  output = D(data)
  test_loss += F.nll_loss(output, target).data[0]
  pred = output.data.max(1)[1] # get the index of the max log-probability
  correct += pred.eq(target.data).cpu().sum()

test_loss /= len(valid_loader) # loss function already averages over batch size
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(valid_loader.dataset),
    100. * correct / len(valid_loader.dataset)))