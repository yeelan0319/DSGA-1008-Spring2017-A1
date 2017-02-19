from __future__ import print_function

import argparse
import glob
import numpy as np
import os
import pickle
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.utils as vutils
from torch.autograd import Variable

G_CKPT_PREFIX = 'unsupervised_G_epoch_'
D_CKPT_PREFIX = 'unsupervised_D_epoch_'
SUPERVISED_CKPT_PREFIX = 'supervised_epoch_'

# Training settings
parser = argparse.ArgumentParser(description='PyTorch semi-supervised MNIST')
# Unsupervised training params
parser.add_argument('--unsupervised-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--unsupervised-epochs', type=int, default=30, metavar='N',
                    help='number of epochs to train DCGAN (default: 25)')
parser.add_argument('--unsupervised-lr', type=float, default=0.0002, metavar='LR',
                    help='learning rate (default: 0.0002)')
# Supervised training params
parser.add_argument('--supervised-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--supervised-epochs', type=int, default=20, metavar='N',
                    help='number of epochs to finetune classifer (default: 10)')
parser.add_argument('--supervised-lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--supervised-momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
# Training monitor params
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--output-interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
# Test params
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
# Continue training and output params
parser.add_argument('--outdir', default='./output', help='folder to put model generate image')
parser.add_argument('--training-index', type=int, default=None,
                    help='Set this number to load and contine train on previous model.')
# Run mode params
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--cuda', action='store_true', help='enables GPU training')
parser.add_argument('--skip-unsupervised-training', action='store_true',
                    help='skip unsupervised training part')
parser.add_argument('--lock-pretrained-params', action='store_true',
                    help='lock pretrained params during supervised training')
parser.add_argument('--minimal-run', action='store_true', help="run minimal version to test code")

# Parse args
args = parser.parse_args()
# Cuda
args.cuda = args.cuda and torch.cuda.is_available()
kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
if args.cuda:
  print("GPU training enabled")
# Other args
random.seed(args.seed)
torch.manual_seed(args.seed)
unsupervised_trained = False
args.unsupervised_training_data = 'data/train_unlabeled.p'
args.supervised_training_data = 'data/train_labeled.p'
args.validation_data = 'data/validation.p'
if args.minimal_run:
  args.unsupervised_epochs = 1
  args.supervised_epochs = 1
  args.unsupervised_training_data = 'data/train_labeled.p'
  args.validation_data = 'data/train_labeled.p'

def get_checkpoint_index(filepath):
  if filepath is None:
    return 0
  else:
    filename = os.path.basename(filepath).split('.')[0]
    return int(filename.split('_')[-1])

def find_latest_ckpt_info(path, type):
  if type == 'unsupervised':
    g_ckpts = sorted(glob.glob(os.path.join(path, G_CKPT_PREFIX + '*')),
      key=get_checkpoint_index)
    g_ckpt = g_ckpts[-1] if len(g_ckpts) > 0 else None
    g_epoch = get_checkpoint_index(g_ckpt)
    print("Found {} generator Checkpoints, return {}".format(
      len(g_ckpts), g_ckpt))
    d_ckpts = sorted(glob.glob(os.path.join(path, D_CKPT_PREFIX + '*')),
      key=get_checkpoint_index)
    d_ckpt = d_ckpts[-1] if len(d_ckpts) > 0 else None
    d_epoch = get_checkpoint_index(d_ckpt)
    print("Found {} discriminitor Checkpoints, return {}".format(
      len(d_ckpts), d_ckpt))
    return min(d_epoch, g_epoch), g_ckpt, d_ckpt
  else:
    ckpts = sorted(glob.glob(os.path.join(path, SUPERVISED_CKPT_PREFIX + '*')),
      key=get_checkpoint_index)
    ckpt = ckpts[-1] if len(ckpts) > 0 else None
    epoch = get_checkpoint_index(ckpt)
    print("Found {} supervised Checkpoints, return {}".format(
      len(ckpts), ckpt))
    return epoch, ckpt

# Make output dir
if not args.training_index:
  args.training_index = len(filter(
    lambda child: os.path.isdir(os.path.join(args.outdir, child)),
    os.listdir(args.outdir))) + 1
args.outdir = os.path.join(args.outdir, 'train {}'.format(args.training_index))
if os.path.exists(args.outdir):
  (latest_unsupervised_epoch, latest_unsupervised_G_ckpt,
    latest_unsupervised_D_ckpt) = find_latest_ckpt_info(args.outdir,
    'unsupervised')
  latest_supervised_epoch, latest_supervised_ckpt = find_latest_ckpt_info(
    args.outdir, 'supervised')
else:
  latest_unsupervised_epoch = 0
  latest_unsupervised_G_ckpt = None
  latest_unsupervised_D_ckpt = None
  latest_supervised_epoch = 0
  latest_supervised_ckpt = None
  os.makedirs(args.outdir)

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

print('Loading data!')
trainset_unlabeled = pickle.load(open(args.unsupervised_training_data, "rb"))
trainset_labeled = pickle.load(open(args.supervised_training_data, "rb"))
validset = pickle.load(open(args.validation_data, "rb"))

unsupervised_loader = torch.utils.data.DataLoader(trainset_unlabeled,
  batch_size=args.unsupervised_batch_size, shuffle=True, **kwargs)
supervised_loader = torch.utils.data.DataLoader(trainset_labeled,
  batch_size=args.supervised_batch_size, shuffle=True, **kwargs)
valid_loader = torch.utils.data.DataLoader(validset,
  batch_size=args.test_batch_size, shuffle=True)

D = DiscriminatorNet() if not args.cuda else DiscriminatorNet().cuda()
D.apply(weights_init)
if latest_unsupervised_D_ckpt is not None:
  print("Loading discriminator model: {} ".format(latest_unsupervised_D_ckpt))
  D.load_state_dict(torch.load(latest_unsupervised_D_ckpt))

############################
# (1) Train DCGAN with unlabeled data
###########################
if args.skip_unsupervised_training or (latest_unsupervised_epoch >= args.unsupervised_epochs):
  print('Skip unsupervised training part')
else:
  print('\n\nTrain DCGAN with 47000 unlabeled data')
  G = GeneratorNet() if not args.cuda else GeneratorNet().cuda()
  G.apply(weights_init)
  if latest_unsupervised_G_ckpt is not None:
    print("Loading generator model: {} ".format(latest_unsupervised_G_ckpt))
    G.load_state_dict(torch.load(latest_unsupervised_G_ckpt))
  D_optimizer = optim.Adam(D.parameters(), lr=args.unsupervised_lr, betas = (0.5, 0.999))
  G_optimizer = optim.Adam(G.parameters(), lr=args.unsupervised_lr, betas = (0.5, 0.999))
  fixed_noise = Variable(torch.randn(1, 20) if not args.cuda else torch.randn(1, 20).cuda())

  for latest_unsupervised_epoch in range(latest_unsupervised_epoch + 1,
    args.unsupervised_epochs + 1):
    for i, (x, _) in enumerate(unsupervised_loader):
      ############################
      # (1.1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
      ###########################
      D_optimizer.zero_grad()
      x = Variable(x if not args.cuda else x.cuda())
      x_output = D(x)
      z = Variable(torch.randn(x.size(0), 20) if not args.cuda else torch.randn(x.size(0), 20).cuda())
      gz = G(z)
      gz_output = D(gz)
      d_loss = -(torch.mean(torch.log(x_output) + torch.log(1 - gz_output)))
      d_loss.backward()
      D_optimizer.step()

      ############################
      # (1.2) Update G network: maximize log(D(G(z)))
      ###########################
      G_optimizer.zero_grad()
      z = Variable(torch.randn(x.size(0), 20) if not args.cuda else torch.randn(x.size(0), 20).cuda())
      gz = G(z)
      gz_output = D(gz)
      g_loss = -torch.mean(torch.log(gz_output))
      g_loss.backward()
      G_optimizer.step()

      if i % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\nLoss_D: {:.4f}\tLoss_G: {:.4f}'.format(
              latest_unsupervised_epoch, i * len(x), len(unsupervised_loader.dataset),
              100. * i / len(unsupervised_loader), d_loss.data[0], g_loss.data[0]))
      if i % args.output_interval == 0:
        vutils.save_image(x.data,'{}/real_samples.png'.format(args.outdir))
        vutils.save_image(gz.data, '{}/fake_samples.png'.format(args.outdir))
        fake = G(fixed_noise)
        vutils.save_image(fake.data,
          '{}/fake_sample_epoch_{}.png'.format(args.outdir, latest_unsupervised_epoch))

    unsupervised_trained = True

  # Save ckpt at last epoch
  if unsupervised_trained:
    torch.save(G.state_dict(), '{}/{}{}.pth'.format(args.outdir, G_CKPT_PREFIX, latest_unsupervised_epoch))
    torch.save(D.state_dict(), '{}/{}{}.pth'.format(args.outdir, D_CKPT_PREFIX, latest_unsupervised_epoch))


############################
# (2) Modify D network: Change the 0-1 discriminator into classifier
###########################
D.classifier = torch.nn.Sequential(
  nn.Linear(800, 10),
  nn.ReLU(inplace=True),
  nn.LogSoftmax()
)


############################
# (3) Update D network with labeled data
###########################
print('\n\nModify discriminator structure for classification')
if args.lock_pretrained_params:
  print("(Lock the pretrained params and only training the classifer)")
  target_params = [
    {'params': list(D.features.parameters())},
    {'params': D.classifier.parameters(), 'lr':args.supervised_lr}
  ]
else:
  target_params = [
    {'params': D.parameters(), 'lr':args.supervised_lr}
  ]
if not unsupervised_trained and latest_supervised_ckpt is not None:
  print("Loading supervised model: {} ".format(latest_supervised_ckpt))
  D.load_state_dict(torch.load(latest_supervised_ckpt))
else:
  latest_supervised_epoch = 0

if latest_supervised_epoch < args.supervised_epochs:
  print('\n\nTuning model with 3000 labeled data')
  optimizer = optim.SGD(target_params, lr=args.supervised_lr,
    momentum=args.supervised_momentum)
  # optimizer = optim.Adam(D.parameters(), lr=args.supervised_lr)
  for latest_supervised_epoch in range(latest_supervised_epoch + 1, args.supervised_epochs + 1):
    for i, (data, target) in enumerate(supervised_loader):
      data = Variable(data if not args.cuda else data.cuda())
      target = Variable(target if not args.cuda else target.cuda())
      optimizer.zero_grad()
      output = D(data)
      loss = F.nll_loss(output, target)
      loss.backward()
      optimizer.step()
      if i % args.log_interval == 0:
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
          latest_supervised_epoch, i * len(data), len(supervised_loader.dataset),
          100. * i / len(supervised_loader), loss.data[0]))

  torch.save(D.state_dict(), '{}/{}{}.pth'.format(args.outdir, SUPERVISED_CKPT_PREFIX, latest_supervised_epoch))

############################
# (4) Evaluate D network with validate data
###########################
print('\n\nEvaluate model with validate labeled data')
D.eval()
test_loss = 0
correct = 0
for data, target in valid_loader:
  data = Variable(data if not args.cuda else data.cuda(), volatile=True)
  target = Variable(target if not args.cuda else data.target())
  output = D(data)
  test_loss += F.nll_loss(output, target).data[0]
  pred = output.data.max(1)[1] # get the index of the max log-probability
  correct += pred.eq(target.data).cpu().sum()

test_loss /= len(valid_loader) # loss function already averages over batch size
print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    test_loss, correct, len(valid_loader.dataset),
    100. * correct / len(valid_loader.dataset)))
