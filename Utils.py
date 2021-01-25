import os
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
  return observation[player][1]+observation[player][2]*-1
  
def getCurrentPlayer(game):
  return 1 if game.current_player==0 else -1
  
def getCheckpointFilename(iteration):
  return 'checkpoint_' + str(iteration) + '.pkl'

def saveTrainExamples(folder, iteration, trainExamples):
  if not os.path.exists(folder):
    os.makedirs(folder)
  filename = os.path.join(folder, getCheckpointFilename(iteration) + ".examples")
  with open(filename, "wb+") as f:
      Pickler(f).dump(trainExamples)
      
class AverageMeter(object):
  """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

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