# searchAgents.py
# ---------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
This file contains all of the agents that can be selected to control Pacman.  To
select an agent, use the '-p' option when running pacman.py.  Arguments can be
passed to your agent using '-a'.  For example, to load a SearchAgent that uses
depth first search (dfs), run the following command:

> python pacman.py -p SearchAgent -a fn=depthFirstSearch

Commands to invoke other search strategies can be found in the project
description.

Please only change the parts of the file you are asked to.  Look for the lines
that say

"*** YOUR CODE HERE ***"

The parts you fill in start about 3/4 of the way down.  Follow the project
description for details.

Good luck and happy searching!
"""

from typing import List, Tuple, Any
from game import Directions
from game import Agent
from game import Actions
import util
import time
import search
import pacman

class GoWestAgent(Agent):
    "An agent that goes West until it can't."

    def getAction(self, state):
        "The agent receives a GameState (defined in pacman.py)."
        if Directions.WEST in state.getLegalPacmanActions():
            return Directions.WEST
        else:
            return Directions.STOP

#######################################################
# This portion is written for you, but will only work #
#       after you fill in parts of search.py          #
#######################################################

class SearchAgent(Agent):
    """
    This very general search agent finds a path using a supplied search
    algorithm for a supplied search problem, then returns actions to follow that
    path.

    As a default, this agent runs DFS on a PositionSearchProblem to find
    location (1,1)

    Options for fn include:
      depthFirstSearch or dfs
      breadthFirstSearch or bfs


    Note: You should NOT change any code in SearchAgent
    """

    def __init__(self, fn='depthFirstSearch', prob='PositionSearchProblem', heuristic='nullHeuristic'):
        # Warning: some advanced Python magic is employed below to find the right functions and problems

        # Get the search function from the name and heuristic
        if fn not in dir(search):
            raise AttributeError(fn + ' is not a search function in search.py.')
        func = getattr(search, fn)
        if 'heuristic' not in func.__code__.co_varnames:
            print('[SearchAgent] using function ' + fn)
            self.searchFunction = func
        else:
            if heuristic in globals().keys():
                heur = globals()[heuristic]
            elif heuristic in dir(search):
                heur = getattr(search, heuristic)
            else:
                raise AttributeError(heuristic + ' is not a function in searchAgents.py or search.py.')
            print('[SearchAgent] using function %s and heuristic %s' % (fn, heuristic))
            # Note: this bit of Python trickery combines the search algorithm and the heuristic
            self.searchFunction = lambda x: func(x, heuristic=heur)

        # Get the search problem type from the name
        if prob not in globals().keys() or not prob.endswith('Problem'):
            raise AttributeError(prob + ' is not a search problem type in SearchAgents.py.')
        self.searchType = globals()[prob]
        print('[SearchAgent] using problem type ' + prob)

    def registerInitialState(self, state):
        """
        This is the first time that the agent sees the layout of the game
        board. Here, we choose a path to the goal. In this phase, the agent
        should compute the path to the goal and store it in a local variable.
        All of the work is done in this method!

        state: a GameState object (pacman.py)
        """
        if self.searchFunction == None: raise Exception("No search function provided for SearchAgent")
        starttime = time.time()
        problem = self.searchType(state) # Makes a new search problem
        self.actions  = self.searchFunction(problem) # Find a path
        if self.actions == None:
            self.actions = []
        totalCost = problem.getCostOfActions(self.actions)
        print('Path found with total cost of %d in %.1f seconds' % (totalCost, time.time() - starttime))
        if '_expanded' in dir(problem): print('Search nodes expanded: %d' % problem._expanded)

    def getAction(self, state):
        """
        Returns the next action in the path chosen earlier (in
        registerInitialState).  Return Directions.STOP if there is no further
        action to take.

        state: a GameState object (pacman.py)
        """
        if 'actionIndex' not in dir(self): self.actionIndex = 0
        i = self.actionIndex
        self.actionIndex += 1
        if i < len(self.actions):
            return self.actions[i]
        else:
            return Directions.STOP

class PositionSearchProblem(search.SearchProblem):
    """
    A search problem defines the state space, start state, goal test, successor
    function and cost function.  This search problem can be used to find paths
    to a particular point on the pacman board.

    The state space consists of (x,y) positions in a pacman game.

    Note: this search problem is fully specified; you should NOT change it.
    """

    def __init__(self, gameState, costFn = lambda x: 1, goal=(1,1), start=None, warn=True, visualize=True):
        """
        Stores the start and goal.

        gameState: A GameState object (pacman.py)
        costFn: A function from a search state (tuple) to a non-negative number
        goal: A position in the gameState
        """
        self.walls = gameState.getWalls()
        self.startState = gameState.getPacmanPosition()
        if start != None: self.startState = start
        self.goal = goal
        self.costFn = costFn
        self.visualize = visualize
        if warn and (gameState.getNumFood() != 1 or not gameState.hasFood(*goal)):
            print('Warning: this does not look like a regular search maze')

        # For display purposes
        self._visited, self._visitedlist, self._expanded = {}, [], 0 # DO NOT CHANGE

    def getStartState(self):
        return self.startState

    def isGoalState(self, state):
        isGoal = state == self.goal

        # For display purposes only
        if isGoal and self.visualize:
            self._visitedlist.append(state)
            import __main__
            if '_display' in dir(__main__):
                if 'drawExpandedCells' in dir(__main__._display): #@UndefinedVariable
                    __main__._display.drawExpandedCells(self._visitedlist) #@UndefinedVariable

        return isGoal

    def getSuccessors(self, state):
        """
        Returns successor states, the actions they require, and a cost of 1.

         As noted in search.py:
             For a given state, this should return a list of triples,
         (successor, action, stepCost), where 'successor' is a
         successor to the current state, 'action' is the action
         required to get there, and 'stepCost' is the incremental
         cost of expanding to that successor
        """

        successors = []
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x,y = state
            dx, dy = Actions.directionToVector(action)
            nextx, nexty = int(x + dx), int(y + dy)
            if not self.walls[nextx][nexty]:
                nextState = (nextx, nexty)
                cost = self.costFn(nextState)
                successors.append( ( nextState, action, cost) )

        # Bookkeeping for display purposes
        self._expanded += 1 # DO NOT CHANGE
        if state not in self._visited:
            self._visited[state] = True
            self._visitedlist.append(state)

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions. If those actions
        include an illegal move, return 999999.
        """
        if actions == None: return 999999
        x,y= self.getStartState()
        cost = 0
        for action in actions:
            # Check figure out the next state and see whether its' legal
            dx, dy = Actions.directionToVector(action)
            x, y = int(x + dx), int(y + dy)
            if self.walls[x][y]: return 999999
            cost += self.costFn((x,y))
        return cost

class StayEastSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the West side of the board.

    The cost function for stepping into a position (x,y) is 1/2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: .5 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn, (1, 1), None, False)

class StayWestSearchAgent(SearchAgent):
    """
    An agent for position search with a cost function that penalizes being in
    positions on the East side of the board.

    The cost function for stepping into a position (x,y) is 2^x.
    """
    def __init__(self):
        self.searchFunction = search.uniformCostSearch
        costFn = lambda pos: 2 ** pos[0]
        self.searchType = lambda state: PositionSearchProblem(state, costFn)

def manhattanHeuristic(position, problem, info={}):
    "The Manhattan distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])

def euclideanHeuristic(position, problem, info={}):
    "The Euclidean distance heuristic for a PositionSearchProblem"
    xy1 = position
    xy2 = problem.goal
    return ( (xy1[0] - xy2[0]) ** 2 + (xy1[1] - xy2[1]) ** 2 ) ** 0.5

#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################

class CornersProblem(search.SearchProblem):
    """
    This search problem finds paths through all four corners of a layout.

    You must select a suitable state space and successor function.
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Initializes the problem by storing walls, Pacman's starting position, and the corners.
        """
        self.walls = startingGameState.getWalls()  # Stores the walls of the maze
        self.startingPosition = startingGameState.getPacmanPosition()  # Stores Pacman's starting position

        # Define the four corners of the maze, ensuring they are inside the boundaries
        top, right = self.walls.height - 2, self.walls.width - 2
        self.corners = ((1, 1), (1, top), (right, 1), (right, top))  

        # Check if there is food in each corner and print a warning if there isn't
        for corner in self.corners:
            if not startingGameState.hasFood(*corner):
                print('Warning: no food in corner ' + str(corner))

        self._expanded = 0  # Counter to track how many search nodes have been expanded

    def getStartState(self):
        """
        Returns the initial state: Pacman's starting position and an empty set of visited corners.
        """
        return (self.startingPosition, tuple())  # State consists of (position, visited_corners)

    def isGoalState(self, state: Any):
        """
        Returns True if all four corners have been visited.
        """
        return len(state[1]) == 4  # Goal is achieved when all four corners are in the visited list

    def getSuccessors(self, state: Any):
        """
        Returns successor states, the actions they require, and a cost of 1.

        Each successor consists of:
        - The new state ((new_x, new_y), updated_corners)
        - The action taken (NORTH, SOUTH, EAST, WEST)
        - The step cost (always 1 in this problem)
        """
        successors = []
        x, y = state[0]  # Extract Pacman current position
        visited_corners = state[1]  # Extract the corners visited so far

        # Try moving in all four possible directions
        for action in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            dx, dy = Actions.directionToVector(action)  # Convert action into movement vector
            nextx, nexty = int(x + dx), int(y + dy)  # Compute the new position

            # Check if the new position is not a wall
            if not self.walls[nextx][nexty]:
                new_corners = visited_corners  # Keep the same visited corners initially
                # If the new position is a corner and hasnt been visited yet, add it to visited corners
                if (nextx, nexty) in self.corners and (nextx, nexty) not in visited_corners:
                    new_corners = visited_corners + ((nextx, nexty),)  # We add the new corner to the tuple
                # Add the successor to the list (new state, action, cost)
                successors.append((( (nextx, nexty), new_corners), action, 1))
        
        self._expanded += 1  # Increment the number of expanded nodes
        return successors  # Return the list of succes

    
    def getCostOfActions(self, actions):
        """
        Returns the cost of a particular sequence of actions.
        If those actions include an illegal move (hitting a wall), return 999999.
        This ensures that illegal moves have an extremely high penalty, making them
        undesirable in the search algorithms.
        """

        # If the action sequence is None (no actions given), returns a very high cost
        if actions is None:
            return 999999

        # Start from Pacman initial position
        x, y = self.startingPosition
        cost = 0  # Initialize cost to 0.

        # Iterate through the sequence of actions
        for action in actions:
            dx, dy = Actions.directionToVector(action)  # Convert action into the movement vector
            x, y = int(x + dx), int(y + dy)  # Compute new position after applying movement
            # If the new position is a wall, return a very high cost to indicate an invalid pth.
            if self.walls[x][y]:
                return 999999
            cost += 1  # Each move has a cost of 1
        return cost  # Return the total cost of the sequence of actions





def cornersHeuristic(state: Any, problem: CornersProblem):
    """
    A heuristic for the CornersProblem.

    This heuristic estimates the cost of reaching all unvisited corners 
    from the current state. It ensures admissibility by always returning 
    a lower bound on the actual shortest path cost.

    Arguments:
      state:   The current search state (position, visited_corners)
      problem: The CornersProblem instance for this layout.

    Returns:
      An estimated cost to reach the goal.
    """
    corners = problem.corners  # Get the corner coordinates
    position, visited_corners = state  # Extract current position and visited corners

    # Identify corners that have not yet been visited
    remaining_corners = [corner for corner in corners if corner not in visited_corners]

    # If all corners are visited, the heuristic cost is 0 (goal state reached)
    if not remaining_corners:
        return 0

    # Initialize the minimum spanning tree (MST) cost estimate
    mst_cost = 0
    current_pos = position  # Start from Pacman's current position
    unvisited_corners = set(remaining_corners)  # Use a set for efficient removal operations

    # Approximate the cost using a greedy approach finnding the nearest corner, movinf to it, and repeat until all corners are visited
    while unvisited_corners:
        # Select the closest unvisited corner based on Manhattan distance
        nearest_corner = min(unvisited_corners, key=lambda c: util.manhattanDistance(current_pos, c))

        # Add the Manhattan distance from the current position to the nearest corner
        mst_cost += util.manhattanDistance(current_pos, nearest_corner)

        # Update the current position to be the nearest corner
        current_pos = nearest_corner

        # Remove the visited corner from the set
        unvisited_corners.remove(nearest_corner)

    # Return the estimated minimum cost required to visit all remaining corners
    return mst_cost




class AStarCornersAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, cornersHeuristic)
        self.searchType = CornersProblem



class FoodSearchProblem:
    """
    A search problem associated with finding a path that collects all the
    food (dots) in a Pacman game.

    A search state in this problem is a tuple (pacmanPosition, foodGrid) where:
      - pacmanPosition: a tuple (x, y) representing Pacman's position.
      - foodGrid:       a Grid (see game.py) where each cell is either True (food exists)
                        or False (food already eaten).
    """

    def __init__(self, startingGameState: pacman.GameState):
        """
        Initializes the problem, storing relevant game information.

        Args:
        - startingGameState: The initial state of the Pacman game.
        """
        self.start = (startingGameState.getPacmanPosition(), startingGameState.getFood())  # Initial state
        self.walls = startingGameState.getWalls()  # Walls grid representation
        self.startingGameState = startingGameState  # Store the full initial game state
        self._expanded = 0  # Tracks the number of expanded nodes (used for debugging/performance analysis)
        self.heuristicInfo = {}  # Dictionary to store reusable information for heuristics

    def getStartState(self):
        """
        Returns the initial state of the search problem.
        This consists of:
        - Pacman's starting position.
        - The initial food grid (indicating which locations contain food).
        """
        return self.start

    def isGoalState(self, state):
        """
        Determines whether the current state is a goal state.
        
        Args:
        - state: A tuple (pacmanPosition, foodGrid).
        
        Returns:
        - True if all food has been eaten (i.e., no True values left in foodGrid).
        """
        return state[1].count() == 0  # If no food remains, the goal has been reached.

    def getSuccessors(self, state):
        """
        Returns successor states, the actions required to reach them, and a step cost.

        Args:
        - state: The current search state, represented as (pacmanPosition, foodGrid).

        Returns:
        - A list of tuples: (successorState, action, stepCost), where:
          - successorState is the new state after moving.
          - action is the direction taken (North, South, East, West).
          - stepCost is always 1 (each move has equal cost).
        """
        successors = []
        self._expanded += 1  # Keep track of how many states have been expanded

        for direction in [Directions.NORTH, Directions.SOUTH, Directions.EAST, Directions.WEST]:
            x, y = state[0]  # Extract Pacman's current position
            dx, dy = Actions.directionToVector(direction)  # Convertthe action to a movment vector
            nextx, nexty = int(x + dx), int(y + dy)  # Compute new position

            # Check if the new position is valid (not a wall)
            if not self.walls[nextx][nexty]:
                nextFood = state[1].copy()  # Copy the current food grid
                nextFood[nextx][nexty] = False  # Mark food as eaten if Pacman moves there
                # Add the successor state: new position, updated food grid, action taken, and cost (the cost is 1)
                successors.append((( (nextx, nexty), nextFood), direction, 1))

        return successors

    def getCostOfActions(self, actions):
        """
        Returns the cost of a given sequence of actions.
        If an action leads into a wall, returns a large cost (999999) to penalize invalid paths.

        Args:
        - actions: A list of Pacman's movement actions.

        Returns:
        - The total cost of executing the given actions.
        """
        if actions is None:
            return 999999  # If no actions are given, return an invalid high cost.

        x, y = self.getStartState()[0]  # Start at Pacmans initial position
        cost = 0  # Initialize cost counter

        for action in actions:
            dx, dy = Actions.directionToVector(action)  # Convert action into movement vector
            x, y = int(x + dx), int(y + dy)  # Compute the next position

            if self.walls[x][y]:  # If the next position is a wall, return a high penalty cost
                return 999999

            cost += 1  # Each step has a cost of 1

        return cost  # Return the total cost for the action sequence



class AStarFoodSearchAgent(SearchAgent):
    "A SearchAgent for FoodSearchProblem using A* and your foodHeuristic"
    def __init__(self):
        self.searchFunction = lambda prob: search.aStarSearch(prob, foodHeuristic)
        self.searchType = FoodSearchProblem


def foodHeuristic(state: Tuple[Tuple, List[List]], problem: FoodSearchProblem):
    """
    Heuristic function for FoodSearchProblem. This heuristic estimates the cost from the current state to the goal state 
    by computing the maximum Manhattan distance between Pacman and the furthest 
    remaining food dot. This ensures an admissible and efficient heuristic.

    Args:
    - state: A tuple (pacmanPosition, foodGrid), where:
        - pacmanPosition: The current position of Pacman as (x, y).
        - foodGrid: A grid of booleans indicating remaining food positions.
    - problem: The FoodSearchProblem instance.
    Returns:
    - An integer representing the heuristic cost estimate.
    """
    position, foodGrid = state  # Extract Pacman current position and the food grid
    food_positions = foodGrid.asList()  # Convert the food grid into a list of food coordinates
    
    if not food_positions:
        return 0  # If there is no food left, return 0 as the goal is already reached
    
    # Compute the maximum Manhattan distance between Pacman and any food dot
    max_distance = max(util.manhattanDistance(position, food) for food in food_positions)
    
    return max_distance  # Return the highest distance as the heuristic estimate



class ClosestDotSearchAgent(SearchAgent):
    """ 
    An agent that searches for all food by repeatedly finding the path 
    to the closest dot using a sequence of searches.
    """

    def registerInitialState(self, state):
        """
        This function initializes the search process when the game starts.
        
        It continuously searches for the closest food dot and follows the path 
        until all food has been collected.
        
        Args:
        - state: The initial GameState (contains Pacman's position, walls, food, etc.).
        """
        self.actions = []  # Stores the sequence of actions that Pacman will follow
        currentState = state  # Keeps track of the current game state

        # Continue searching until all food is eaten
        while currentState.getFood().count() > 0:
            nextPathSegment = self.findPathToClosestDot(currentState)  # Find path to nearest food
            self.actions += nextPathSegment  # Append the path found to the overall action list
            
            # Validate the path and update the game state step by step
            for action in nextPathSegment:
                legal = currentState.getLegalActions()  # Get possible legal moves from the current state
                if action not in legal:
                    t = (str(action), str(currentState))
                    raise Exception('findPathToClosestDot returned an illegal move: %s!\n%s' % t)
                
                currentState = currentState.generateSuccessor(0, action)  # Apply action to move Pacman

        self.actionIndex = 0  # Reset action index to start following the path
        print('Path found with cost %d.' % len(self.actions))  # Display total cost of the found path

    def findPathToClosestDot(self, gameState: pacman.GameState):
        """
        Finds the shortest path to the closest food dot from the current state.

        Uses BFS (Breadth-First Search) to find the shortest path.

        Args:
        - gameState: The current GameState.

        Returns:
        - A list of actions leading Pacman to the nearest food.
        """
        problem = AnyFoodSearchProblem(gameState)  # Create a search problem for finding the closest food
        return search.bfs(problem)  # Use BFS to find the optimal path to the closest food dot


class AnyFoodSearchProblem(PositionSearchProblem):
    """
    A specialized search problem for finding a path to any food dot.

    This problem inherits from PositionSearchProblem but modifies the goal state 
    to stop when Pacman reaches the closest food.
    """

    def __init__(self, gameState):
        """
        Initializes the problem based on the given game state.

        Args:
        - gameState: The current game state.
        """
        self.food = gameState.getFood()  # Store the food grid
        self.walls = gameState.getWalls()  # Store the wall layout
        self.startState = gameState.getPacmanPosition()  # Get Pacman's starting position
        self.costFn = lambda x: 1  # Set the cost of moving to any state as 1 (uniform cost)
        self._visited, self._visitedlist, self._expanded = {}, [], 0  # For debugging purposes

    def isGoalState(self, state: Tuple[int, int]):
        """
        Determines whether the given state is a goal state.

        The goal is reached when Pacman reaches any tile that contains food.

        Args:
        - state: A tuple (x, y) representing Pacman's current position.

        Returns:
        - True if the state contains food, otherwise False.
        """
        return self.food[state[0]][state[1]]  # Check if there is food at this position


def mazeDistance(point1: Tuple[int, int], point2: Tuple[int, int], gameState: pacman.GameState) -> int:
    """
    Computes the shortest path distance between two points in the maze.

    This function uses BFS to find the shortest path between two points while avoiding walls.

    Args:
    - point1: The starting point (x1, y1).
    - point2: The destination point (x2, y2).
    - gameState: The GameState object (walls and layout of the maze).

    Returns:
    - The number of steps in the shortest path from point1 to point2.
    """
    x1, y1 = point1
    x2, y2 = point2
    walls = gameState.getWalls()

    # Ensure both points are valid (not inside a wall)
    assert not walls[x1][y1], 'point1 is a wall: ' + str(point1)
    assert not walls[x2][y2], 'point2 is a wall: ' + str(point2)

    # Create a search problem to find the shortest path between the two points
    prob = PositionSearchProblem(gameState, start=point1, goal=point2, warn=False, visualize=False)
    
    return len(search.bfs(prob))  # Return the number of steps in the shortest BFS path
