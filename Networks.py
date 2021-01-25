import os
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from Utils import AverageMeter

class Connect4Net(nn.Module):
  
  def __init__(self, args):
    # Parameters
    self.rows = args['rows']
    self.cols = args['cols']
    self.num_actions = args['num_actions']
    self.num_channels = args['num_channels']
    self.dropout = args['dropout']
    # Internal structures
    super(Connect4Net, self).__init__()
    # Convolution layers
    self.conv1 = nn.Conv2d(1, self.num_channels, 3, stride=1, padding=1)
    self.conv2 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1, padding=1)
    self.conv3 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
    self.conv4 = nn.Conv2d(self.num_channels, self.num_channels, 3, stride=1)
    # Batch Normalization layers
    self.bn1 = nn.BatchNorm2d(self.num_channels)
    self.bn2 = nn.BatchNorm2d(self.num_channels)
    self.bn3 = nn.BatchNorm2d(self.num_channels)
    self.bn4 = nn.BatchNorm2d(self.num_channels)
    # Fully connected layers
    self.fc1 = nn.Linear(self.num_channels*(self.cols-4)*(self.rows-4), 1024)
    self.fc_bn1 = nn.BatchNorm1d(1024)
    self.fc2 = nn.Linear(1024, 512)
    self.fc_bn2 = nn.BatchNorm1d(512)
    self.fc3 = nn.Linear(512, self.num_actions)
    self.fc4 = nn.Linear(512, 1)

  def forward(self, s):
    #                                                            s: batch_size x cols x rows
    s = s.view(-1, 1, self.cols, self.rows)                      # batch_size x 1 x cols x rows
    s = F.relu(self.bn1(self.conv1(s)))                          # batch_size x num_channels x cols x rows
    s = F.relu(self.bn2(self.conv2(s)))                          # batch_size x num_channels x cols x rows
    s = F.relu(self.bn3(self.conv3(s)))                          # batch_size x num_channels x (cols-2) x (rows-2)
    s = F.relu(self.bn4(self.conv4(s)))                          # batch_size x num_channels x (cols-4) x (rows-4)
    s = s.view(-1, self.num_channels*(self.cols-4)*(self.rows-4))
    s = F.dropout(F.relu(self.fc_bn1(self.fc1(s))), p=self.dropout, training=self.training)  # batch_size x 1024
    s = F.dropout(F.relu(self.fc_bn2(self.fc2(s))), p=self.dropout, training=self.training)  # batch_size x 512
    pi = self.fc3(s)                                                                         # batch_size x num_actions
    v = self.fc4(s)                                                                          # batch_size x 1
    return F.log_softmax(pi, dim=1), torch.tanh(v)
    
class Connect4NetWrapper():
  
  def __init__(self, args):
    self.rows = args['rows']
    self.cols = args['cols']
    self.num_actions = args['num_actions']
    self.cuda = args['cuda']
    self.epochs = args['epochs']
    self.batch_size = args['batch_size']
    # Network
    self.net = Connect4Net(args)
    # Use CUDA?
    if self.cuda:
      self.net.cuda()
 
  def predict(self, board):
    board = torch.FloatTensor(board.astype(np.float64))
    if self.cuda: 
      board = board.contiguous().cuda()
    board = board.view(1, self.rows, self.cols)
    self.net.eval()
    with torch.no_grad():
        pi, v = self.net(board)
    return torch.exp(pi).data.cpu().numpy()[0], v.data.cpu().numpy()[0]
    
  def train(self, examples):
    optimizer = optim.Adam(self.net.parameters()) # TODO

    for epoch in range(self.epochs):
      print('EPOCH ::: ' + str(epoch + 1))
      self.net.train()
      pi_losses = AverageMeter()
      v_losses = AverageMeter()

      batch_count = int(len(examples) / self.batch_size)

      t = tqdm(range(batch_count), desc='Training Net')
      for _ in t:
        sample_ids = np.random.randint(len(examples), size=self.batch_size)
        boards, pis, vs = list(zip(*[examples[i] for i in sample_ids]))
        boards = torch.FloatTensor(np.array(boards).astype(np.float64))
        target_pis = torch.FloatTensor(np.array(pis))
        target_vs = torch.FloatTensor(np.array(vs).astype(np.float64))

        # predict
        if self.cuda:
          boards, target_pis, target_vs = boards.contiguous().cuda(), target_pis.contiguous().cuda(), target_vs.contiguous().cuda()

        # compute output
        out_pi, out_v = self.net(boards)
        l_pi = self.loss_pi(target_pis, out_pi)
        l_v = self.loss_v(target_vs, out_v)
        total_loss = l_pi + l_v

        # record loss
        pi_losses.update(l_pi.item(), boards.size(0))
        v_losses.update(l_v.item(), boards.size(0))
        t.set_postfix(Loss_pi=pi_losses, Loss_v=v_losses)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
  def loss_pi(self, targets, outputs):
    return -torch.sum(targets * outputs) / targets.size()[0]

  def loss_v(self, targets, outputs):
    return torch.sum((targets - outputs.view(-1)) ** 2) / targets.size()[0]
    
  def save_checkpoint(self, folder='checkpoint', filename='checkpoint.net.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(folder):
        os.mkdir(folder)
    torch.save({'state_dict': self.net.state_dict()}, filepath)

  def load_checkpoint(self, folder='checkpoint', filename='checkpoint.net.tar'):
    filepath = os.path.join(folder, filename)
    if not os.path.exists(filepath):
        raise ("No model in path {}".format(filepath))
    map_location = None if self.cuda else 'cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    self.net.load_state_dict(checkpoint['state_dict'])