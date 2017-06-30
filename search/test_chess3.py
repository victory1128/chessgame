#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

from mcts.mcts import MCTS
from mcts.tree_policies import UCB1, Go
from mcts.default_policies import immediate_reward
from mcts.backups import monte_carlo
from mcts.default_policies import RandomKStepRollOut, RandomKStepRollOut_Value
from mcts.graph import StateNode
import numpy as np
from chess.chess_state import ChessState
from chess.pvr_network import policy_nn, rollout_nn, value_nn
import cProfile, pstats, io, time





def new_chess_game():
    x=np.zeros([10,9])
    x[0,:]=np.array([4,3,5,6,7,6,5,3,4])
    x[3,:]=np.array([1,0,1,0,1,0,1,0,1])
    x[2,1]=2
    x[2,7]=2
    x=x-np.flipud(x)
    return np.int32(x)
move = [[0,6,2,4],
        [7,7,7,3],
        [3,6,4,6],
        [9,7,7,6],
        [0,7,2,6],
        [9,8,9,7],
        [0,8,0,7],
        [9,7,3,7],
        [3,2,4,2],
        [9,1,7,2],
        [2,6,4,5], #bad
        [3,7,3,4],
        [0,1,2,2],
        [3,4,3,8],
        [4,5,5,3],
        [3,8,3,1],
        [2,1,7,1],
        [3,1,7,1],
        [2,7,2,6],
        [9,2,7,4],
        [0,5,1,4],
        [9,0,8,0],
        [2,2,3,4],
        [8,0,8,5],
        [5,3,7,2],
        [7,1,7,2],
        [0,0,0,1],
        [7,2,8,2],
        [0,1,7,1],
        [8,2,8,3],
        [0,7,7,7],
        [8,5,3,5],
        [7,1,7,3],  #bad
        [8,3,7,3],
        [7,7,7,6],
        [3,5,3,4],
        [7,6,6,6],
        [7,3,1,3],
        [2,4,0,6],
        [3,4,3,6],
        [2,6,2,1],
        [3,6,0,6],
        [1,4,0,5],
        [6,4,5,4],
        [2,1,9,1],
        [9,3,8,4],
        [6,6,6,2],
        [5,4,4,4],
        [6,2,9,2],
        [8,4,9,3],
        [9,2,9,3],
        [9,4,9,3]]


pr = cProfile.Profile()
pr.enable()

pos = new_chess_game()

for i in range(len(move)):
    movei = move[i]
    if pos[movei[0],movei[1]] != 0:
        pos[movei[2],movei[3]] = pos[movei[0],movei[1]]
        pos[movei[0],movei[1]] = 0
    else:
        ValueErr("error")

print(np.flipud(pos))



mcts = MCTS(tree_policy=Go(c=5), 
            default_policy=RandomKStepRollOut_Value(20, 0.95),
            backup=monte_carlo)

policy_fun = policy_nn()
rollout_fun = rollout_nn()
value_fun = value_nn() 

root = StateNode(None, ChessState(pos, 1, policy_fun, rollout_fun, value_fun, False ))
best_action = mcts(root, n=500)

pr.disable()
s = io.StringIO()
sortby = 'cumulative'
ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
ps.print_stats()
print(s.getvalue())

print(best_action.action)


