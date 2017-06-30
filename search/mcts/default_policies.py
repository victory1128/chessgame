#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: "Johannes Kulick"
@modification: YongfengLi
"""


import random
import itertools
import bisect


def immediate_reward(state_node):
    """
    Estimate the reward with the immediate return of that state.
    :param state_node:
    :return:
    """
    return state_node.state.reward(state_node.parent.parent.state,
                                   state_node.parent.action)


class RandomKStepRollOut(object):
    """
    Estimate the reward with the sum of returns of a k step rollout
    """
    def __init__(self, k):
        self.k = k

    def __call__(self, state_node):
        self.current_k = 0

        def stop_k_step(state):
            self.current_k += 1
            return self.current_k > self.k or state.is_terminal()

        return _roll_out(state_node, stop_k_step)


class RandomKStepRollOut_Value(object):
    """
    Estimate the reward with the sum of returns of a k step rollout
    """
    def __init__(self, k, lam):
        self.k = k
        self.lam = lam

    def __call__(self, state_node):
        self.current_k = 0

        def stop_k_step(state):
            self.current_k += 1
            return self.current_k > self.k or state.is_terminal()

        return _roll_out2(state_node, stop_k_step)*self.lam + (1-self.lam)*state_node.state.v


def random_terminal_roll_out(state_node):
    """
    Estimate the reward with the sum of a rollout till a terminal state.
    Typical for terminal-only-reward situations such as games with no
    evaluation of the board as reward.

    :param state_node:
    :return:
    """
    def stop_terminal(state):
        return state.is_terminal()

    return _roll_out(state_node, stop_terminal)


def _roll_out(state_node, stopping_criterion):
    reward = 0
    state = state_node.state
    parent = state_node.parent.parent.state
    action = state_node.parent.action
    while not stopping_criterion(state):
        reward += state.reward(parent, action)
       # action = random.choice(state.actions) # wrong
        cumdist = list(itertools.accumulate(state.actions_p))
        x = random.random()*cumdist[-1]
        inx = bisect.bisect(cumdist,x)
        action = state.actions[inx]
        parent = state
        state = parent.real_world_perform(action) #change


    return reward


def _roll_out2(state_node, stopping_criterion):
    reward = 0
    state = state_node.state
    parent = state_node.parent.parent.state
    action = state_node.parent.action
    while not stopping_criterion(state):
        reward += state.reward(parent, action)
       # action = random.choice(state.actions) # wrong
        cumdist = list(itertools.accumulate(state.actions_p+0.01))
        x = random.random()*cumdist[-1]
        inx = bisect.bisect(cumdist,x)
        action = state.actions[inx]
        parent = state
        state = parent.real_world_perform(action) #change
    
    if not state.is_terminal():
        reward = state.vf(state.pos)

    return reward


