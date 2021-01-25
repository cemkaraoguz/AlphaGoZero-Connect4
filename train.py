import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import deque
from Networks import Connect4NetWrapper
from TreeSearch import MCTS
from Utils import getCurrentPlayer, getCanonicalForm, saveTrainExamples, loadTrainExamples
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
    'numIters': 1000,                     # Number of iterations
    'numEps': 100,                        # Number of complete self-play games to simulate during a new iteration
    'epochs': 10,                         # Number of learning epochs
    'batch_size': 64,                     # Batch size for training
    # MCTS
    'numMCTSSims': 25,                    # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'tempThreshold': 0,
    'maxlenQueue': 200000,                # Max number of game examples acquired from self plays.
    'maxItersForTrainExamplesHist': 20,   # Size of buffer for total training samples in terms of iterations
    'checkpointFolder': "./data",
    'checkpointLoadIteration': 147,
  }
  
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  connect4net = Connect4NetWrapper(args)
  # Check if we continue from a checkpoint
  if args['checkpointLoadIteration']>0:
    trainExamplesHistory = loadTrainExamples(args['checkpointFolder'], args['checkpointLoadIteration'])
    connect4net.load_checkpoint(folder=args['checkpointFolder'])
    iteration_start = args['checkpointLoadIteration']+1
  else:
    trainExamplesHistory = []
    iteration_start = 1
    
  for i in range(iteration_start, args['numIters'] + 1): 
    
    print("Iteration:", i)
    
    trainExamplesFromSelfPlay = deque([], maxlen=args['maxlenQueue'])
    for _ in tqdm(range(args['numEps']), desc="Self Play"):
      # Execute self plays
      mcts = MCTS(connect4net, args)
      trainExamplesFromSelfPlay += executeEpisode(game, mcts, args['tempThreshold'])
    # Save trajectories from self-plays for training
    trainExamplesHistory.append(trainExamplesFromSelfPlay)
    if len(trainExamplesHistory) > args['maxItersForTrainExamplesHist']:
      trainExamplesHistory.pop(0)
    # Save checkpoint
    saveTrainExamples(args['checkpointFolder'], i-1, trainExamplesHistory)
    connect4net.save_checkpoint(folder=args['checkpointFolder'], filename='checkpoint.net.tar')
    # shuffle examples before training
    trainExamples = []
    for e in trainExamplesHistory:
        trainExamples.extend(e)
    shuffle(trainExamples)
    # Training
    connect4net.train(trainExamples)
    
  game.close()
  