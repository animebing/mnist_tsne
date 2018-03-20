from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.autograd import Variable
import os
import numpy as np

# Training settings
parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_dir', type=str, default='./train',
                    help="the path to save the trained model")
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args.batch_size, shuffle=True, **kwargs)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv1_bn = nn.BatchNorm2d(10)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_bn = nn.BatchNorm2d(20)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc1_bn = nn.BatchNorm1d(50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        bn_1 = self.conv1_bn(self.conv1(x))
        x = F.relu(F.max_pool2d(bn_1, 2))
        #x = F.relu(F.max_pool2d(self.conv1(x), 2))
        bn_2 = self.conv2_bn(self.conv2(x))
        #x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(bn_2), 2))
        x = x.view(-1, 320)
        bn_fc = self.fc1_bn(self.fc1(x))
        #x = F.relu(self.fc1(x))
        x = F.relu(bn_fc)
        x = F.dropout(x, training=self.training)
        #x = self.fc2(x)
        #return F.log_softmax(x, dim=1)
        return x

model = Net()
model_path = os.path.join(args.save_dir, 'mnist_bn.pth')
model.load_state_dict(torch.load(model_path))

if args.cuda:
    model.cuda()

def generate_feature():
    model.eval()
    cnt = 0
    out_target = []
    out_data = []
    out_output =[]
    for data, target in train_loader:
        cnt += len(data)
        print("processing: %d/%d" % (cnt, len(train_loader.dataset)))
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        output_np = output.data.cpu().numpy()
        target_np = target.data.cpu().numpy()
        data_np = data.data.cpu().numpy()

        out_output.append(output_np)
        out_target.append(target_np[:, np.newaxis])
        out_data.append(np.squeeze(data_np))


    output_array = np.concatenate(out_output, axis=0)
    target_array = np.concatenate(out_target, axis=0)
    data_array = np.concatenate(out_data, axis=0)

    np.save(os.path.join(args.save_dir, 'output.npy'), output_array, allow_pickle=False)
    np.save(os.path.join(args.save_dir, 'target.npy'), target_array, allow_pickle=False)
    np.save(os.path.join(args.save_dir, 'data.npy'), data_array, allow_pickle=False)

generate_feature()