import numpy as np
from tqdm import tqdm
from random import shuffle
from collections import deque
from Networks import Connect4NetWrapper
from TreeSearch import MCTS
from Utils import getCurrentPlayer, getStateRepresentation
from Utils import saveTrainExamples, loadTrainExamples
from Utils import loadLogData, saveLogData, prepareTrainingData
from evaluate import evaluate
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
    state = getStateRepresentation(game)
    temp = int(episodeStep < tempThreshold)
    pi = mcts.getActionProb(game, temp=temp)
    trainExamples.append([state, currentPlayer, pi, None])
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
    'numIters': 200,                      # Number of iterations
    'numEps': 100,                        # Number of complete self-play games to simulate during a new iteration
    'epochs': 10,                         # Number of learning epochs
    'batch_size': 64,                     # Batch size for training
    'doStateAggregation': True,
    # MCTS
    'numMCTSSims': 50,                    # Number of games moves for MCTS to simulate.
    'cpuct': 4,                           # Upper confidence bound parameter
    'tempThreshold': 15,                  # Temperature for action selection
    'doScaleReward': True,                # Scale reward w.r.t game length?
    'w_noise': 0.5,                       # Weight of Dirichlet noise added to the priors in the root node of MCTS
    'alpha': 0.5,                         # Dirichlet noise parameter
    'maxlenQueue': 200000,                # Max number of game examples acquired from self plays.
    'maxItersForTrainExamplesHist': 20,   # Size of buffer for total training samples in terms of iterations
    'checkpointFolder': "./data",
    'checkpointLoadIteration': 0,
    'num_tests': 100,
    'comments': "AdamW+Dirichlet+State Aggregation+Variable reward",
  }
  
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  
  # Check if we continue from a checkpoint
  if args['checkpointLoadIteration']>0:
    trainExamplesHistory = loadTrainExamples(args['checkpointFolder'], args['checkpointLoadIteration'])
    log = loadLogData(folder=args['checkpointFolder'])
    args_train = log['args'].copy()
    connect4net = Connect4NetWrapper(args_train)
    connect4net.load_checkpoint(folder=args['checkpointFolder'])
    iteration_start = args['checkpointLoadIteration']+1
  else:
    args_train = args.copy()
    connect4net = Connect4NetWrapper(args_train)
    trainExamplesHistory = []
    iteration_start = 1
    log = {}
    log['args'] = args_train

  args_test = args_train.copy()
  args_test['w_noise'] = 0.0

  for i in range(iteration_start, args_train['numIters'] + 1): 
    
    print("Iteration:", i)
    
    trainExamplesFromSelfPlay = deque([], maxlen=args_train['maxlenQueue'])
    for _ in tqdm(range(args_train['numEps']), desc="Self Play"):
      # Execute self plays
      mcts = MCTS(connect4net, args_train)
      trainExamplesFromSelfPlay += executeEpisode(game, mcts, args_train['tempThreshold'])
    # Save trajectories from self-plays for training
    trainExamplesHistory.append(trainExamplesFromSelfPlay)
    if len(trainExamplesHistory) > args_train['maxItersForTrainExamplesHist']:
      trainExamplesHistory.pop(0)
    # Save checkpoint
    saveTrainExamples(args_train['checkpointFolder'], i-1, trainExamplesHistory)
    connect4net.save_checkpoint(folder=args_train['checkpointFolder'], filename='checkpoint.net.tar')
    # prepare examples before training
    if args_train['doStateAggregation']:
      trainExamples = prepareTrainingData(trainExamplesHistory) 
    else:
      trainExamples = []
      for e in trainExamplesHistory:
        trainExamples.extend(e)
      shuffle(trainExamples)
    # Training
    pi_losses, v_losses = connect4net.train(trainExamples)
    # Testing
    num_wins, num_draws = evaluate(args_test, connect4net, opponent="OSLA")
    print("Win rate : {}% Draw rate : {}%".format(round(num_wins*100/args_train['num_tests']), round(num_draws*100/args_train['num_tests'])))
    # Logging
    if i not in log.keys():
      log[i] = {}
    log['last_iteration'] = i
    log[i]['pi_losses'] = pi_losses.avg
    log[i]['v_losses'] = v_losses.avg
    log[i]['num_wins'] = num_wins
    log[i]['num_draws'] = num_draws
    saveLogData(log, args_train['checkpointFolder'])
    
  game.close()
  