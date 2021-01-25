import numpy as np
from TreeSearch import MCTS

class AlphaZeroAgent():
  
  def __init__(self, nnetwrapper, args):
    self.nnetwrapper = nnetwrapper
    self.args = args
    self.name = "AlphaZero"
    
  def selectAction(self, game):
    print("alpha plays")
    pi = self.mcts.getActionProb(game, temp=0)
    return np.random.choice(len(pi), p=pi)
    
  def reset(self):
    self.mcts = MCTS(self.nnetwrapper, self.args)
    
class RandomPlayAgent():

  def __init__(self):
    self.name = "Random play"
    
  def selectAction(self, game):
    return np.random.choice(game.get_moves())
    
  def reset(self):
    pass
