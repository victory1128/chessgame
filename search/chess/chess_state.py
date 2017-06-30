#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

import numpy as np
from . import ChessStep2 as cs


class ChessAction(object):
    def __init__(self, action, p):
        self.action = action.copy()
        self._hash = 1000*action[0] + 100*action[1] + 10*action[2] + action[3]
        self.p = p

    def __hash__(self):
        return int(self._hash)

    def __eq__(self, other):
        return (self.action == other.action).all()

    def __str__(self):
        return str(self.action)

    def __repr__(self):
        return str(self.action)

class ChessState(object):
    def __init__(self, pos, player, policyfun, rolloutfun, valuefun, isfast):
        self.pos = pos.copy()
        self.player = player
        self.pf = policyfun
        self.rf = rolloutfun
        self.vf = valuefun

        if isfast:
            actf = self.rf
        else:
            actf = self.pf
            self.v = (valuefun(pos)*2-1)#*0.8+0.2/20*cs.game_value(pos)

        if self.player > 0:
            actions, actions_p = actf(pos)
        else:
            actions, actions_p = actf(-np.flipud(pos))
            if actions.shape[0] > 0:
                actions[:,0] = 9 - actions[:,0]
                actions[:,2] = 9 - actions[:,2]
        
        self.actions = []
        if actions.shape[0] > 0:
            actions_p = actions_p[0]
            self.actions_p = actions_p
            for i in range((actions.shape)[0]):
                self.actions.append(ChessAction(actions[i,:], actions_p[i]))
        else:
            self.actions_p = actions_p
   
   
    def perform(self, action):
        # build next state
        pos = self.pos.copy()
        player = self.player
        act = action.action
        if self.pos[act[0]][act[1]] != 0:
            pos[act[2]][act[3]] = pos[act[0]][act[1]]
            pos[act[0]][act[1]] = 0
        else:
            raise ValueError("There is no chess in %d %d",act[0],act[1])
        
        return ChessState(pos, -player, self.pf, self.rf, self.vf,  False)


    def real_world_perform(self, action):
        # build next state
        pos = self.pos.copy()
        player = self.player
        act = action.action
        if self.pos[act[0]][act[1]] != 0:
            pos[act[2]][act[3]] = pos[act[0]][act[1]]
            pos[act[0]][act[1]] = 0
        else:
            raise ValueError("There is no chess in %d %d",act[0],act[1])
        
        return ChessState(pos, -player, self.pf, self.rf, self.vf, True)



    def reward(self, parent, action):
        pos = parent.pos.copy() 
        player = parent.player

        if len(parent.actions) == 0:
            if player == 1:
                return -1
            else:
                return 1

        act = action.action
        if pos[act[0]][act[1]] != 0:
            pos[act[2]][act[3]] = pos[act[0]][act[1]]
            pos[act[0]][act[1]] = 0
        else:
            raise ValueError("There is no chess in %d %d",act[0],act[1])

        if (7 not in pos):
            return -1
        elif (-7 not in pos):
            return 1
        else:
            return 0


    def is_terminal(self):
        if (-7 not in self.pos) or (7 not in self.pos) or len(self.actions)==0:
            return True
        else:
            return False
            
    def __eq__(self, other):
        return (self.pos == other.pos).all()

    def __hash__(self):
        return hash(tuple(self.pos.reshape(1,90)[0]))








