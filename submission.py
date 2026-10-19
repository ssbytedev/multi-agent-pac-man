from util import manhattanDistance
from game import Directions
import random, util

from game import Agent
# BEGIN_HIDE
# END_HIDE

class ReflexAgent(Agent):
  """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
  """
  def __init__(self):
    self.lastPositions = []
    self.dc = None


  def getAction(self, gameState):
    """
    getAction chooses among the best options according to the evaluation function.

    getAction takes a GameState and returns some Directions.X for some X in the set {North, South, West, East, Stop}
    ------------------------------------------------------------------------------
    Description of GameState and helper functions:

    A GameState specifies the full game state, including the food, capsules,
    agent configurations and score changes. In this function, the |gameState| argument
    is an object of GameState class. Following are a few of the helper methods that you
    can use to query a GameState object to gather information about the present state
    of Pac-Man, the ghosts and the maze.

    gameState.getLegalActions():
        Returns the legal actions for the agent specified. Returns Pac-Man's legal moves by default.

    gameState.generateSuccessor(agentIndex, action):
        Returns the successor state after the specified agent takes the action.
        Pac-Man is always agent 0.

    gameState.getPacmanState():
        Returns an AgentState object for pacman (in game.py)
        state.configuration.pos gives the current position
        state.getDirection() gives the travel vector

    gameState.getGhostStates():
        Returns list of AgentState objects for the ghosts

    gameState.getNumAgents():
        Returns the total number of agents in the game

    gameState.getScore():
        Returns the score corresponding to the current state of the game


    The GameState class is defined in pacman.py and you might want to look into that for
    other helper methods, though you don't need to.
    """
    # Collect legal moves and successor states
    legalMoves = gameState.getLegalActions()

    # Choose one of the best actions
    scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
    bestScore = max(scores)
    bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
    chosenIndex = random.choice(bestIndices) # Pick randomly among the best

    # BEGIN_HIDE
    # END_HIDE

    return legalMoves[chosenIndex]

  def evaluationFunction(self, currentGameState, action):
    """
    The evaluation function takes in the current and proposed successor
    GameStates (pacman.py) and returns a number, where higher numbers are better.

    The code below extracts some useful information from the state, like the
    remaining food (oldFood) and Pacman position after moving (newPos).
    newScaredTimes holds the number of moves that each ghost will remain
    scared because of Pacman having eaten a power pellet.
    """
    # Useful information you can extract from a GameState (pacman.py)
    successorGameState = currentGameState.generatePacmanSuccessor(action)
    newPos = successorGameState.getPacmanPosition()
    oldFood = currentGameState.getFood()
    newGhostStates = successorGameState.getGhostStates()
    newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

    # BEGIN_HIDE
    # END_HIDE
    return successorGameState.getScore()


def scoreEvaluationFunction(currentGameState):
  """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the Pacman GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
  """
  return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
  """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available to
    the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

    You do not need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.
    It's only partially specified, and designed to be extended.  Agent
    (game.py) is another abstract class.
  """

  def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
    self.index = 0 # Pacman is always agent index 0
    self.evaluationFunction = util.lookup(evalFn, globals())
    self.depth = int(depth)

######################################################################################
# Problem 1b: implementing minimax

class MinimaxAgent(MultiAgentSearchAgent):
  """
    Your minimax agent (problem 1)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action from the current gameState using self.depth
      and self.evaluationFunction. Terminal states can be found by one of the following:
      pacman won, pacman lost or there are no legal moves.

      Here are some method calls that might be useful when implementing minimax.

      gameState.getLegalActions(agentIndex):
        Returns a list of legal actions for an agent
        agentIndex=0 means Pacman, ghosts are >= 1

      gameState.generateSuccessor(agentIndex, action):
        Returns the successor game state after an agent takes an action

      gameState.getNumAgents():
        Returns the total number of agents in the game

      gameState.getScore():
        Returns the score corresponding to the current state of the game

      gameState.isWin():
        Returns True if it's a winning state

      gameState.isLose():
        Returns True if it's a losing state

      self.depth:
        The depth to which search should continue

    """
    # ### START CODE HERE ###
    numAgents = gameState.getNumAgents()

    def minimax(state, depth, agentIndex):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      if not legalActions:
        return self.evaluationFunction(state)

      nextAgent = (agentIndex + 1) % numAgents
      nextDepth = depth - 1 if nextAgent == 0 else depth

      if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          value = max(value, minimax(successor, nextDepth, nextAgent))
        return value

      value = float('inf')
      for action in legalActions:
        successor = state.generateSuccessor(agentIndex, action)
        value = min(value, minimax(successor, nextDepth, nextAgent))
      return value

    legalActions = gameState.getLegalActions(0)
    if not legalActions:
      return Directions.STOP

    bestAction = legalActions[0]
    bestValue = float('-inf')

    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      nextAgent = 1 % numAgents
      nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
      value = minimax(successor, nextDepth, nextAgent)

      if value > bestValue:
        bestValue = value
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 2a: implementing alpha-beta

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
    Your minimax agent with alpha-beta pruning (problem 2)
  """

  def getAction(self, gameState):
    """
      Returns the minimax action using self.depth and self.evaluationFunction
    """
    # ### START CODE HERE ###
    numAgents = gameState.getNumAgents()

    def alphabeta(state, depth, agentIndex, alpha, beta):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      if not legalActions:
        return self.evaluationFunction(state)

      nextAgent = (agentIndex + 1) % numAgents
      nextDepth = depth - 1 if nextAgent == 0 else depth

      if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          value = max(
            value,
            alphabeta(successor, nextDepth, nextAgent, alpha, beta)
          )
          if value > beta:
            return value
          alpha = max(alpha, value)
        return value

      value = float('inf')
      for action in legalActions:
        successor = state.generateSuccessor(agentIndex, action)
        value = min(
          value,
          alphabeta(successor, nextDepth, nextAgent, alpha, beta)
        )
        if value < alpha:
          return value
        beta = min(beta, value)
      return value

    legalActions = gameState.getLegalActions(0)
    if not legalActions:
      return Directions.STOP

    bestAction = legalActions[0]
    alpha = float('-inf')
    beta = float('inf')

    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      nextAgent = 1 % numAgents
      nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
      value = alphabeta(successor, nextDepth, nextAgent, alpha, beta)

      if value > alpha:
        alpha = value
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 3b: implementing expectimax

class ExpectimaxAgent(MultiAgentSearchAgent):
  """
    Your expectimax agent (problem 3)
  """

  def getAction(self, gameState):
    """
      Returns the expectimax action using self.depth and self.evaluationFunction

      All ghosts should be modeled as choosing uniformly at random from their
      legal moves.
    """
    # ### START CODE HERE ###
    numAgents = gameState.getNumAgents()

    def expectimax(state, depth, agentIndex):
      if state.isWin() or state.isLose() or depth == 0:
        return self.evaluationFunction(state)

      legalActions = state.getLegalActions(agentIndex)
      if not legalActions:
        return self.evaluationFunction(state)

      nextAgent = (agentIndex + 1) % numAgents
      nextDepth = depth - 1 if nextAgent == 0 else depth

      if agentIndex == 0:
        value = float('-inf')
        for action in legalActions:
          successor = state.generateSuccessor(agentIndex, action)
          value = max(
            value,
            expectimax(successor, nextDepth, nextAgent)
          )
        return value

      total = 0.0
      for action in legalActions:
        successor = state.generateSuccessor(agentIndex, action)
        total += expectimax(successor, nextDepth, nextAgent)

      return total / len(legalActions)

    legalActions = gameState.getLegalActions(0)
    if not legalActions:
      return Directions.STOP

    bestAction = legalActions[0]
    bestValue = float('-inf')

    for action in legalActions:
      successor = gameState.generateSuccessor(0, action)
      nextAgent = 1 % numAgents
      nextDepth = self.depth - 1 if nextAgent == 0 else self.depth
      value = expectimax(successor, nextDepth, nextAgent)

      if value > bestValue:
        bestValue = value
        bestAction = action

    return bestAction
    # ### END CODE HERE ###

######################################################################################
# Problem 4a (extra credit): creating a better evaluation function

def betterEvaluationFunction(currentGameState):
  """
    Your extreme, unstoppable evaluation function (problem 4).

    DESCRIPTION: A weighted linear combination of features, starting from the
    game score. Terminal states return finite values so expectimax can still
    weigh risky moves by probability. Remaining food is penalized, and the
    reciprocal distance to the nearest pellet pulls Pac-Man toward food,
    except while a catchable scared ghost exists, when only the ghost matters.
    Remaining capsules carry a large penalty (also applied to winning states)
    so Pac-Man uses every capsule before finishing the level. Scared ghosts
    that can be reached before their timer expires get a strong reciprocal-
    distance reward, since each one eaten is worth +200 points. Non-scared
    ghosts within one step get a large penalty, with a smaller penalty at
    distance two.
  """
  # ### START CODE HERE ###
  capsules = currentGameState.getCapsules()

  if currentGameState.isWin():
    return currentGameState.getScore() + 1000 - 50.0 * len(capsules)
  if currentGameState.isLose():
    return currentGameState.getScore() - 1000

  pos = currentGameState.getPacmanPosition()
  food = currentGameState.getFood().asList()
  ghosts = currentGameState.getGhostStates()

  score = currentGameState.getScore()

  # Is there a scared ghost that can still be caught before it recovers?
  hunting = any(
    g.scaredTimer > manhattanDistance(pos, g.getPosition())
    for g in ghosts if g.getPosition() is not None
  )

  # Food: fewer pellets left is better; only chase food when not hunting
  score -= 4.0 * len(food)
  if food and not hunting:
    nearestFood = min(manhattanDistance(pos, f) for f in food)
    score += 10.0 / (nearestFood + 1)

  # Capsules: strongly prefer to use them (enables +200 ghost eats)
  score -= 50.0 * len(capsules)
  if capsules:
    nearestCapsule = min(manhattanDistance(pos, c) for c in capsules)
    score += 5.0 / (nearestCapsule + 1)

  # Ghosts: hunt reachable scared ghosts, avoid dangerous ones
  for ghost in ghosts:
    ghostPos = ghost.getPosition()
    if ghostPos is None:
      continue

    distance = manhattanDistance(pos, ghostPos)

    if ghost.scaredTimer > distance:
      score += 300.0 / (distance + 1)
    elif ghost.scaredTimer > 0:
      score += 20.0 / (distance + 1)
    else:
      if distance <= 1:
        score -= 200
      elif distance <= 2:
        score -= 60.0 / distance
      else:
        score -= 2.0 / distance

  return score
  # ### END CODE HERE ###

# Abbreviation
better = betterEvaluationFunction
