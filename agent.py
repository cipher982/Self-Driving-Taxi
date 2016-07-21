import random
from environment import Agent, Environment
from planner import RoutePlanner
from simulator import Simulator
import pandas as pd
import re

class LearningAgent(Agent):
    """An agent that learns to drive in the smartcab world."""

    def __init__(self, env):
        super(LearningAgent, self).__init__(env)  # sets self.env = env, state = None, next_waypoint = None, and a default color
        self.color = 'red'  # override color
        self.planner = RoutePlanner(self.env, self)  # simple route planner to get next_waypoint
        self.qs = {}
        self.time = 0
        self.errors = 0
        self.possible_actions = (None, 'left', 'forward', 'right')
        self.optimal_val = 0

    def reset(self, destination=None):
        self.planner.route_to(destination)
        # TODO: Prepare for a new trip; reset any variables here, if required
        # Not sure what to put here???

    def best_action(self, state):
        """
        Returns the best action (the one with the maximum Q-value)
        or one of the best actions, given a state.
        """        
        # get all possible q-values for the state
        all_qs = {action: self.qs.get((state, action), 0) for action in self.possible_actions}        
        
        # pick the actions that yield the largest q-value for the state
        optimal_actions = [action for action in self.possible_actions if all_qs[action] == max(all_qs.values())]
#        self.optimal = random.choice(max(all_qs.values()))
        self.optimal_val = float(max(all_qs.values()))
        # return one of the best actions at random in case of a tie
        return random.choice(optimal_actions)        

    def update(self, t):
        # Gather inputs
        self.next_waypoint = self.planner.next_waypoint()  # from route planner, also displayed by simulator
        inputs = self.env.sense(self)
        deadline = self.env.get_deadline(self)
        
        # Added a time/learning rate to improve upon the models performance
        # Starts out learning quickly then decreases over the time incremented
        self.time += 1
        alpha = 1.0 / self.time
        gamma = .5

        # At first I had included 'right', but after more thinking, realized it is not needed
        self.state = (inputs['light'], inputs['oncoming'], inputs['left'], self.next_waypoint)

        # Pick the best known action
        action = self.best_action(self.state)
        
        # Grab action and get the reward
        reward = self.env.act(self, action)
		
		# Adding new line to grab errors and record them
        if reward < 0:
            self.errors += reward

        # Update the q-value of the (state, action) pair
        self.qs[(self.state, action)] = (1 - alpha) * self.qs.get((self.state, action), 0) \
        + alpha * (reward + gamma * self.optimal_val)
            
        print "Reward is"
        print reward
        print "LearningAgent.update(): deadline = {}, inputs = {}, action = {}, reward = {}".format(deadline, inputs, action, reward)  # [debug]

def run():
    """Run the agent for a finite number of trials."""
    total_rew = []
    # Set up environment and agent
    e = Environment()  # create environment (also adds some dummy traffic)
    a = e.create_agent(LearningAgent)  # create agent
    e.set_primary_agent(a, enforce_deadline=True)  # specify agent to track
    # NOTE: You can set enforce_deadline=False while debugging to allow longer trials

    # Now simulate it
    sim = Simulator(e, update_delay=0, display=False)  # create simulator (uses pygame when display=True, if available)
    # NOTE: To speed up simulation, reduce update_delay and/or set display=False
#    for x in range(20):
    sim.run(n_trials=100)  # run for a specified number of trials
    # NOTE: To quit midway, press Esc or close pygame window, or hit Ctrl+C on the command-line


if __name__ == '__main__':
    run()