import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import deque
from Networks import Connect4NetWrapper
from TreeSearch import MCTS
from Utils import getCurrentPlayer, getCanonicalForm, saveTrainExamples
import torch
import gym
import gym_connect4
from colorama import init
init()

def executeEpisode(game, mcts, tempThreshold):
  trainExamples = []
  observation = game.reset()  
  currentPlayer = getCurrentPlayer(game)
  episodeStep = 0
  done = False
  while not done:
    episodeStep += 1
    canonicalBoard = getCanonicalForm(game)
    temp = int(episodeStep < tempThreshold)
    pi = mcts.getActionProb(game, temp=temp)
    trainExamples.append([canonicalBoard, currentPlayer, pi, None])
    '''
    sym = self.game.getSymmetries(canonicalBoard, pi)
    for b, p in sym:
        trainExamples.append([b, self.currentPlayer, p, None])
    '''
    action = np.random.choice(len(pi), p=pi)
    observation, reward, done, info = game.step(action)
    currentPlayer = getCurrentPlayer(game)
    if done:
      return [(x[0], x[2], reward[game.current_player] * ((-1) ** (x[1] != currentPlayer))) for x in trainExamples]



if __name__=="__main__":
  
  args = {
    # Game
    'cols': 7,
    'rows': 6,
    'num_actions': 7,
    # NN
    'num_channels': 512,
    'dropout': 0.3,
    'cuda': torch.cuda.is_available(),
    # Training
    'numIters': 1000,
    'numEps': 100,
    'numMCTSSims': 25,
    'cpuct': 1,
    'tempThreshold': 0,
    'maxlenQueue': 200000,                # Max number of game examples to train the neural networks.
    'checkpointFolder': "./data",
    'epochs': 10,
    'batch_size': 64,
    'numItersForTrainExamplesHistory': 20,
  }
  
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  connect4net = Connect4NetWrapper(args)
  
  trainExamplesHistory = []
  for i in range(1, args['numIters'] + 1): 
    trainExamplesFromSelfPlay = deque([], maxlen=args['maxlenQueue'])
    for _ in tqdm(range(args['numEps']), desc="Self Play"):
      mcts = MCTS(connect4net, args)
      trainExamplesFromSelfPlay += executeEpisode(game, mcts, args['tempThreshold'])
    trainExamplesHistory.append(trainExamplesFromSelfPlay)
    if len(trainExamplesHistory) > args['numItersForTrainExamplesHistory']:
      trainExamplesHistory.pop(0)
    
    saveTrainExamples(args['checkpointFolder'], i-1, trainExamplesHistory)
    connect4net.save_checkpoint(folder=args['checkpointFolder'], filename='checkpoint.net.tar')
    
    # shuffle examples before training
    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
    
    connect4net.train(trainExamples)

  game.close()
  