import numpy as np
from TreeSearch import MCTS

class PlayAgent():
  
  def __init__(self):
    self.name = "Abstract player class"
  
  def selectAction(self, game):
    pass
  
  def reset(self):
    pass
  
class AlphaZeroAgent(PlayAgent):
  
  def __init__(self, nnetwrapper, args):
    self.nnetwrapper = nnetwrapper
    self.args = args
    self.name = "AlphaZero"
    
  def selectAction(self, game):
    pi = self.mcts.getActionProb(game, temp=0)
    return np.random.choice(len(pi), p=pi)
    
  def reset(self):
    self.mcts = MCTS(self.nnetwrapper, self.args)
    
class RandomPlayAgent(PlayAgent):

  def __init__(self):
    self.name = "Random play"
    
  def selectAction(self, game):
    return np.random.choice(game.get_moves())

class HumanAgent(PlayAgent):

  def __init__(self):
    self.name = "carbon based life form"
    
  def selectAction(self, game):
    available_actions = game.get_moves()
    while True:
      print("Available columns are ", available_actions)
      action = input("Please enter your move:")
      if(not action.isdigit() or int(action) not in available_actions):
        print("Action should be one of the positive integer values of the available columns!")
      else:
        return int(action)
