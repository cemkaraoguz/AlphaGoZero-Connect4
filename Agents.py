import numpy as np
from TreeSearch import MCTS
from Utils import getCanonicalForm, getStateRepresentation

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
    if self.args['numMCTSSims']==0:
      state = getStateRepresentation(game)
      pi, v = self.nnetwrapper.predict(state)
      valids = np.zeros(self.args['num_actions'])
      valids[game.get_moves()]=1
      pi = pi * valids  # masking invalid moves
      # renormalize
      sum_pi = np.sum(pi) 
      if sum_pi>0:
        pi /= sum_pi  
      else:
        pi += valids
        pi /= np.sum(pi)
    else:
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
        
class OSLAAgent(PlayAgent):
  
  def __init__(self, inarow=4):
    self.name = "One Step Look Ahead"
    self.inarow = inarow
    
  def selectAction(self, game):
    # Get list of valid moves
    available_actions = game.get_moves()
    # Get state info
    canonicalBoard = getCanonicalForm(game)
    # Use the heuristic to score each possible board state in the next turn
    scores = dict(zip(available_actions, [self.getActionScore(canonicalBoard, col, 1) for col in available_actions]))
    # Get actions that maximize the heuristic score
    max_cols = [key for key in scores.keys() if scores[key] == max(scores.values())]
    # Select random action from the maximizing actions
    return int(np.random.choice(max_cols))
  
  def getActionScore(self, canonicalBoard, col, mark):
    '''
    Calculates score if agent drops piece in selected column
    '''
    next_state = self.simulateAction(canonicalBoard, col, mark)
    score = self.getHeuristic(next_state, mark)
    return score

  def simulateAction(self, canonicalBoard, col, mark):
    '''
    Returns the next board state for the simulated action
    '''
    next_state = canonicalBoard.copy()
    for row in range(np.shape(canonicalBoard)[0]-1, -1, -1):
      if next_state[row][col] == 0:
        break
    next_state[row][col] = mark     
    return next_state

  def getHeuristic(self, canonicalBoard, mark):
    '''
    Assigns a score for a window of 4 using the following heuristics:
    3 own piece in a window of 4 = +1
    4 own piece in a window of 4 = +1000000
    3 opponent pieces in a window of 4 = -100
    Otherwise = 0
    Returns the sum of all window scores
    '''
    num_discs = [3,4,3]
    piece = [mark, mark, -1*mark]
    num_windows = self.countWindows(canonicalBoard, num_discs, piece)
    score = num_windows[0] + 1e6*num_windows[1] - 1e2*num_windows[2]
    return score
      
  def countWindows(self, canonicalBoard, num_discs, piece):
    '''
    Analyses all possible vertical/horizontal/diagonal windows of 4
    '''
    rows, columns = np.shape(canonicalBoard)
    nHeuristics = len(num_discs)
    num_windows = np.zeros(nHeuristics)
    # horizontal
    for row in range(rows):
      for col in range(columns-(self.inarow-1)):
        window = list(canonicalBoard[row, col:col+self.inarow])
        for i in range(nHeuristics):
          if self.check_window(window, num_discs[i], piece[i]):
            num_windows[i] += 1
    # vertical
    for row in range(rows-(self.inarow-1)):
      for col in range(columns):
        window = list(canonicalBoard[row:row+self.inarow, col])
        for i in range(nHeuristics):
          if self.check_window(window, num_discs[i], piece[i]):
            num_windows[i] += 1
    # positive diagonal
    for row in range(rows-(self.inarow-1)):
      for col in range(columns-(self.inarow-1)):
        window = list(canonicalBoard[range(row, row+self.inarow), range(col, col+self.inarow)])
        for i in range(nHeuristics):
          if self.check_window(window, num_discs[i], piece[i]):
            num_windows[i] += 1
    # negative diagonal
    for row in range(self.inarow-1, rows):
      for col in range(columns-(self.inarow-1)):
        window = list(canonicalBoard[range(row, row-self.inarow, -1), range(col, col+self.inarow)])
        for i in range(nHeuristics):
          if self.check_window(window, num_discs[i], piece[i]):
            num_windows[i] += 1
    return num_windows

  def check_window(self, window, num_discs, piece):
    '''
    Checks if a given window satisfies heuristic conditions
    '''
    return (window.count(piece) == num_discs and window.count(0) == self.inarow-num_discs)
      