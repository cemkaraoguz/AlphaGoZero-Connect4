import torch
import gym
import gym_connect4
import random
import numpy as np
from tqdm import tqdm
from Networks import Connect4NetWrapper
from Utils import getCurrentPlayer, getCanonicalForm, getValueFromDict
from Agents import AlphaZeroAgent, RandomPlayAgent, OSLAAgent

def executeTest(game, agents):
  agents[0].reset()
  agents[1].reset()
  observation = game.reset()
  currentPlayer = game.current_player
  episodeStep = 0
  done = False
  while not done:
    episodeStep += 1
    canonicalBoard = getCanonicalForm(game)
    action = agents[currentPlayer].selectAction(game)    
    observation, reward, done, info = game.step(action)
    currentPlayer = game.current_player
  return game.winner
  
def evaluate(args, net=None, opponent="random"):
  num_tests = getValueFromDict(args, 'num_tests', 100)
  # Set up a game
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  # Set up the network
  if net is None:
    connect4net = Connect4NetWrapper(args)    
    connect4net.load_checkpoint(folder=args['checkpointFolder'])
  else:
    connect4net = net
  # Set up the agent
  agent_alphazero = AlphaZeroAgent(connect4net, args)
  # Set up the challenger
  if opponent=="random":
    agent_opponent = RandomPlayAgent()
  elif opponent=="OSLA":
    agent_opponent = OSLAAgent()
  else:
    raise NotImplementedError
  agents = [agent_alphazero, agent_opponent]
  # Run tests
  num_wins = np.zeros(2)
  nTests_half = num_tests//2
  t = tqdm(range(nTests_half), desc='Evaluation 1/2')
  for _ in t:
    winner = executeTest(game, agents)
    if winner==-1:
      #print("Draw!")
      pass
    else:
      num_wins[winner]+=1
  # Reverse the agents
  agents.reverse()
  num_wins = np.flip(num_wins)
  t = tqdm(range(num_tests-nTests_half), desc='Evaluation 2/2')
  for _ in t:
    winner = executeTest(game, agents)
    if winner==-1:
      #print("Draw!")
      pass
    else:
      num_wins[winner]+=1
  return num_wins[1], num_tests-num_wins.sum()

if __name__=="__main__":

  num_tests = 100
  opponent = "OSLA"
  args = {
    # Game
    'cols': 7,
    'rows': 6,
    'num_actions': 7,
    # MCTS
    'numMCTSSims': 25,                    # Number of games moves for MCTS to simulate.
    'cpuct': 4,
    'tempThreshold': 0,    
    # NN
    'num_channels': 512,
    'dropout': 0.3,
    'cuda': torch.cuda.is_available(),
    'checkpointFolder': "./data",
    'num_tests': num_tests,
  }
  
  num_wins, num_draws = evaluate(args, net=None, opponent=opponent)
  
  print()
  print("AlphaZero win rate : {}%".format(round(num_wins*100/num_tests)))
  print("{} agent win rate : {}%".format(opponent, round((num_tests-num_wins-num_draws)*100/num_tests)))
  print("Draw rate : {}".format(round(num_draws*100/num_tests)))
  
