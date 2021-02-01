from Utils import getValueFromDict
from Utils import getStateRepresentation, isGameEnded, getPlayLength
import math
import numpy as np

EPS = 1e-8

class MCTS():
  """
  This class implements MCTS.
  Adapted from https://github.com/suragnair/alpha-zero-general
  """
  
  def __init__(self, nnet, args):
    self.nnet = nnet
    self.numMCTSSims = getValueFromDict(args, 'numMCTSSims')
    self.cpuct = getValueFromDict(args, 'cpuct')
    self.num_actions = getValueFromDict(args, 'num_actions')
    self.doScaleReward = getValueFromDict(args, 'doScaleReward', False)
    self.w_noise = getValueFromDict(args, 'w_noise')
    assert(self.w_noise>=0 and self.w_noise<=1)
    self.alpha = getValueFromDict(args, 'alpha')
    self.game = None
    self.Qsa = {}  # stores Q values for s,a (as defined in the paper)
    self.Nsa = {}  # stores #times edge s,a was visited
    self.Ns = {}   # stores #times board s was visited
    self.Ps = {}   # stores initial policy (returned by neural net)
    self.Es = {}   # stores game.getGameEnded ended for board s
    self.Vs = {}   # stores game.getValidMoves for board s

  def getActionProb(self, game, temp=1):
    """
    This function performs numMCTSSims simulations of MCTS starting from
    canonicalBoard.
    Returns:
        probs: a policy vector where the probability of the ith action is
               proportional to Nsa[(s,a)]**(1./temp)
    """    
    for i in range(self.numMCTSSims):
      self.game = game.clone()      
      self.search(isRootNode=True)
      
    state = getStateRepresentation(game)
    s = state.tostring() # Root node
    counts = [self.Nsa[(s, a)] if (s, a) in self.Nsa else 0 for a in range(self.num_actions)]
    if temp == 0:
      bestAs = np.array(np.argwhere(counts == np.max(counts))).flatten()
      bestA = np.random.choice(bestAs)
      probs = [0] * len(counts)
      probs[bestA] = 1
      return probs
    else:
      counts = [x ** (1. / temp) for x in counts]
      counts_sum = float(sum(counts))
      probs = [x / counts_sum for x in counts]
      return probs

  def search(self, isRootNode):
      """
      This function performs one iteration of MCTS. It is recursively called
      till a leaf node is found. The action chosen at each node is one that
      has the maximum upper confidence bound as in the paper.
      Once a leaf node is found, the neural network is called to return an
      initial policy P and a value v for the state. This value is propagated
      up the search path. In case the leaf node is a terminal state, the
      outcome is propagated up the search path. The values of Ns, Nsa, Qsa are
      updated.
      NOTE: the return values are the negative of the value of the current
      state. This is done since v is in [-1,1] and if v is the value of a
      state for the current player, then its value is -v for the other player.
      Returns:
          v: the negative of the value of the current canonicalBoard
      """
      state = getStateRepresentation(self.game)
      s = state.tostring()

      if s not in self.Es:
        self.Es[s] = isGameEnded(self.game)
      if self.Es[s] != 0:
        # terminal node
        if self.doScaleReward and self.Es[s]>0: # TODO: check + or -
          # Scale the reward to a value between 0.1 and 1.0 proportional to game 1/game length
          reward_scaler = ((-1*getPlayLength(self.game))+46)/39
        else:
          reward_scaler = 1.0
        return -self.Es[s]*reward_scaler

      if s not in self.Ps:
        # leaf node
        self.Ps[s], v = self.nnet.predict(state)
        if isRootNode:
          self.Ps[s] = (1.0-self.w_noise)*self.Ps[s] + self.w_noise*np.random.dirichlet(np.full(self.num_actions, self.alpha))
        valids = np.zeros(self.num_actions)
        valids[self.game.get_moves()]=1
        self.Ps[s] = self.Ps[s] * valids  # masking invalid moves
        sum_Ps_s = np.sum(self.Ps[s]) 
        if sum_Ps_s > 0:
          self.Ps[s] /= sum_Ps_s  # renormalize
        else:
          # if all valid moves were masked make all valid moves equally probable
          # NB! All valid moves may be masked if either your NNet architecture is insufficient or you've get overfitting or something else.
          # If you have got dozens or hundreds of these messages you should pay attention to your NNet and/or training process.   
          self.Ps[s] = self.Ps[s] + valids
          self.Ps[s] /= np.sum(self.Ps[s])
        self.Vs[s] = valids
        self.Ns[s] = 0
        return -v

      valids = self.Vs[s]
      cur_best = -float('inf')
      best_act = -1
      # pick the action with the highest upper confidence bound
      for a in range(self.num_actions):
        if valids[a]:
          if (s, a) in self.Qsa:
            u = self.Qsa[(s, a)] + self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
          else:
            u = self.cpuct * self.Ps[s][a] * math.sqrt(self.Ns[s] + EPS)  # Q = 0 ?
          if u > cur_best:
            cur_best = u
            best_act = a

      a = best_act
      observation, reward, done, info = self.game.step(a)      
      v = self.search(isRootNode=False)

      if (s, a) in self.Qsa:
        self.Qsa[(s, a)] = (self.Nsa[(s, a)] * self.Qsa[(s, a)] + v) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
      else:
        self.Qsa[(s, a)] = v
        self.Nsa[(s, a)] = 1

      self.Ns[s] += 1
      return -v
