import torch
import gym
import gym_connect4
import random
import numpy as np
from tqdm import tqdm
from Networks import Connect4NetWrapper
from Utils import getCurrentPlayer, getCanonicalForm
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

if __name__=="__main__":

  args = {
    # Game
    'cols': 7,
    'rows': 6,
    'num_actions': 7,
    # MCTS
    'numMCTSSims': 25,                    # Number of games moves for MCTS to simulate.
    'cpuct': 1,
    'tempThreshold': 0,    
    # NN
    'num_channels': 512,
    'dropout': 0.3,
    'cuda': torch.cuda.is_available(),
    'checkpointFolder': "./data",
  }
  
  nTests = 100
  
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  connect4net = Connect4NetWrapper(args)    
  connect4net.load_checkpoint(folder=args['checkpointFolder'])
  agent_alphazero = AlphaZeroAgent(connect4net, args)
  agent_random = RandomPlayAgent()
  agent_osla = OSLAAgent()
  # Select agents to play
  #agents = [agent_alphazero, agent_random]
  agents = [agent_alphazero, agent_osla]
  # Run tests
  num_wins = np.zeros(2)
  nTests_half = nTests//2
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
  t = tqdm(range(nTests-nTests_half), desc='Evaluation 2/2')
  for _ in t:
    winner = executeTest(game, agents)
    if winner==-1:
      #print("Draw!")
      pass
    else:
      num_wins[winner]+=1
  
  print()
  for i, agent in enumerate(agents):
    print("Agent {} win rate : {}%".format(agent.name, round(num_wins[i]*100/nTests)))
  print("Draw rate : {}".format(round((nTests-num_wins.sum())*100/nTests)))
  
