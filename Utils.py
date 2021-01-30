import os
import numpy as np
from pickle import Pickler, Unpickler

def isGameEnded(game):
  # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
  if game.winner is None:
    return 0
  elif game.winner==-1:
    return -1             # Draw
  elif game.winner==0:
    return 1
  elif game.winner==1:
    return -1
  else:
    # Not supposed to be here
    raise ValueError('invalid winner id:'+str(game.winner))
    
def getCanonicalForm(game):
  observation = game.get_player_observations()
  player = game.current_player
  return np.transpose(np.flip(observation[player][1]+observation[player][2]*-1))
  
def getStateRepresentation(game):
  observation = game.get_player_observations()
  player = game.current_player
  return np.array([np.transpose(np.flip(observation[player][1])), 
    np.transpose(np.flip(observation[player][2]))])
  
def getCurrentPlayer(game):
  return 1 if game.current_player==0 else -1
  
def getPlayLength(game):
  return np.sum(game.board>=0)
  
def getCheckpointFilename(iteration):
  return 'checkpoint_' + str(iteration) + '.pkl'

def saveTrainExamples(folder, iteration, trainExamples):
  if not os.path.exists(folder):
    os.makedirs(folder)
  filename = os.path.join(folder, getCheckpointFilename(iteration) + ".examples")
  with open(filename, "wb+") as f:
      Pickler(f).dump(trainExamples)

def loadTrainExamples(folder, iteration):
  examplesFile = os.path.join(folder, getCheckpointFilename(iteration) + ".examples")
  if not os.path.isfile(examplesFile):
     raise FileNotFoundError("file does not exist {}".format(examplesFile))
  else:
    print("Loading training examples {}".format(examplesFile))
    with open(examplesFile, "rb") as f:
      return Unpickler(f).load()

def saveLogData(logdata, folder):
  filepath = os.path.join(folder, 'logdata.pkl')
  if not os.path.exists(folder):
      os.mkdir(folder)
  with open(filepath, "wb+") as f:
    Pickler(f).dump(logdata)

def loadLogData(folder):
  filepath = os.path.join(folder, 'logdata.pkl')
  if not os.path.exists(filepath):
      raise FileNotFoundError("No log data in path {}".format(filepath))
  else:
    print("Loading log data {}".format(filepath))
    with open(filepath, "rb") as f:
      return Unpickler(f).load()
    
def getValueFromDict(indict, key, defaultVal=None):
  if key in indict.keys():
    return indict[key]
  else:
    if defaultVal is None:
      raise KeyError
    else:
      return defaultVal
    
class AverageMeter(object):
  '''
  From https://github.com/pytorch/examples/blob/master/imagenet/main.py
  '''

  def __init__(self):
    self.val = 0
    self.avg = 0
    self.sum = 0
    self.count = 0

  def __repr__(self):
    return f'{self.avg:.2e}'

  def update(self, val, n=1):
    self.val = val
    self.sum += val * n
    self.count += n
    self.avg = self.sum / self.count