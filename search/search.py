# search.py
# ---------
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
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util
from game import Directions
from typing import List
from util import PriorityQueue, Stack, Queue

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()




def tinyMazeSearch(problem: SearchProblem) -> List[Directions]:
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """Perform depth-first search to find a solution to the given search problem."""

    # Initialize the stack (fringe) for DFS
    fringe = Stack()
    # Push the start state onto the stack with an empty action list
    fringe.push((problem.getStartState(), []))
    # Set to keep track of visited states
    visited = set()

    # Continue until there are no more states to explore
    while not fringe.isEmpty():
        # Pop the last state and its associated actions from the stack
        state, actions = fringe.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(state):
            return actions  # Return the actions if the goal state is reached

        # If the state has not been visited yet
        if state not in visited:
            visited.add(state)  # Mark the state as visited
            # Explore the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                new_actions = actions + [action]  # Update the action list with the new action
                # Push the successor state and the updated actions onto the stack
                fringe.push((successor, new_actions))

    return []  # Return an empty list if no solution is found

def breadthFirstSearch(problem):
    """Perform breadth-first search to find the shortest path to the goal state."""

    # Initialize the queue (fringe) for BFS
    fringe = Queue()
    # Enqueue the start state with an empty action list
    fringe.push((problem.getStartState(), []))
    # Set for visited states
    visited = set()

    while not fringe.isEmpty():
        # Dequeue the front state
        state, actions = fringe.pop()

        # Check if the current state is the goal state to return the actions
        if problem.isGoalState(state):
            return actions 

        # If the state has not been visited yet
        if state not in visited:
            visited.add(state) 
            # Explore the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                new_actions = actions + [action]  # Update the action list with the new action
                # Enqueue the successor state and the updated actions
                fringe.push((successor, new_actions))

    return []  

def uniformCostSearch(problem):
    """Perform uniform cost search to find the least costly path to the goal state."""

    # Initialize the priority queue (fringe) for UCS
    fringe = PriorityQueue()
    # Push the start state with a cost of 0
    fringe.push((problem.getStartState(), []), 0)
    # Set for visited states
    visited = set()

    while not fringe.isEmpty():
        # Pop the state with the lowest cost from the priority queue
        state, actions = fringe.pop()

        # Check if the current state is the goal state
        if problem.isGoalState(state):
            return actions  # Return the actions if the goal state is reached

        # If the state has not been visited yet
        if state not in visited:
            visited.add(state) 
            # Explore the successors of the current state
            for successor, action, stepCost in problem.getSuccessors(state):
                new_actions = actions + [action]  # Update the action list
                new_cost = problem.getCostOfActions(new_actions)  # Calculate the total cost

                # Push the successor state and the updated actions with their cost
                fringe.push((successor, new_actions), new_cost)

    return []  

def nullHeuristic(state, problem=None) -> float:
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem: SearchProblem, heuristic=nullHeuristic) -> List[Directions]:
    """Search the node that has the lowest combined cost and heuristic first."""
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
