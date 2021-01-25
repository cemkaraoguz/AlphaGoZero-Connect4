import torch
import gym
import gym_connect4
import random
from Networks import Connect4NetWrapper
from Utils import getCurrentPlayer, getCanonicalForm
from Agents import AlphaZeroAgent, RandomPlayAgent

def executeGame(game, agents):
  agents[0].reset()
  agents[1].reset()
  observation = game.reset()
  currentPlayer = game.current_player
  episodeStep = 0
  done = False
  while not done:
    episodeStep += 1
    canonicalBoard = getCanonicalForm(game)
    game.render()
    action = agents[currentPlayer].selectAction(game)
    
    print("player {} {} takes action {}".format(currentPlayer, agents[currentPlayer].name, action))
    
    observation, reward, done, info = game.step(action)
    currentPlayer = game.current_player
    if done:
      game.render()
      print("Game Over")
      if game.winner==-1:
        print("Draw!")
      else:
        print("player {} {} wins!".format(game.winner, agents[game.winner].name))
      
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
  
  game = gym.make("Connect4-v0", width=args['cols'], height=args['rows'])
  connect4net = Connect4NetWrapper(args)    
  connect4net.load_checkpoint(folder=args['checkpointFolder'])
  agent_alphazero = AlphaZeroAgent(connect4net, args)
  agent_random = RandomPlayAgent()
  # Select agents to play
  agents = [agent_alphazero, agent_random]
  # Shuffle agents to randomize starting agent
  random.shuffle(agents)
  # Play the game
  executeGame(game, agents)
  
  
