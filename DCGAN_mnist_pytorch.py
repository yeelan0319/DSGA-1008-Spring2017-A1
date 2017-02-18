from __future__ import print_function
import pickle
import numpy as np
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable


# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
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
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

print('loading data!')
trainset_unlabeled = pickle.load(open("data/train_labeled.p", "rb"))
train_loader = torch.utils.data.DataLoader(trainset_unlabeled, batch_size=64,
  shuffle=True, **kwargs)


class Discriminator(nn.Module):
  def __init__(self):
    super(Discriminator, self).__init__()
    self.conv1 = nn.Conv2d(1, 16, kernel_size=4, stride=2)
    self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
    self.fc = nn.Linear(800, 1)

  def forward(self, x):
    # 28 * 28 * 1
    x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
    # 13 * 13 * 16
    x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
    # 5 * 5 * 32
    x = x.view(-1, 800)
    # 800 * 1
    x = F.sigmoid(self.fc(x)) 
    return x


class Generator(nn.Module):
  def __init__(self):
    super(Generator, self).__init__()
    self.fc1 = nn.Linear(20, 800)
    self.convTrans1 = nn.ConvTranspose2d(32, 16, kernel_size=5, stride=2)
    self.convTrans2 = nn.ConvTranspose2d(16, 1, kernel_size=4, stride=2)

  def forward(self, z):
    # 20 * 1
    x = F.relu(self.fc1(z))
    # 800 * 1
    x = x.view(-1, 32, 5, 5)
    # 5 * 5 * 32
    x = F.relu(self.convTrans1(x))
    # 13 * 13 * 16
    x = F.tanh(self.convTrans2(x))
    # 28 * 28 * 1
    return x

D = Discriminator()
G = Generator()

D_optimizer = optim.Adam(D.parameters(), lr=args.lr)
G_optimizer = optim.Adam(G.parameters(), lr=args.lr)


def train(epoch):
  D.train()
  G.train()
  for batch_idx, (x, _) in enumerate(train_loader):
    # Train D
    D_optimizer.zero_grad()
    x = Variable(x)
    x_output = D(x)
    z = Variable(torch.randn(x.size(0), 20))
    gz = G(z)
    gz_output = D(gz)
    d_loss = -(torch.mean(torch.log(x_output) + torch.log(1 - gz_output)))
    d_loss.backward()
    D_optimizer.step()
    # Train G
    G_optimizer.zero_grad()
    z = Variable(torch.randn(x.size(0), 20))
    gz = G(z)
    gz_output = D(gz)
    g_loss = -torch.mean(torch.log(gz_output))
    g_loss.backward()
    G_optimizer.step()
    if batch_idx % args.log_interval == 0:
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, batch_idx * len(x), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), d_loss.data[0]))

for epoch in range(1, args.epochs + 1):
  train(epoch)
