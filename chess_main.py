#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

from search.mcts.mcts import MCTS, myMCTS
from search.mcts.tree_policies import UCB1, Go
from search.mcts.default_policies import immediate_reward
from search.mcts.backups import monte_carlo
from search.mcts.default_policies import RandomKStepRollOut, RandomKStepRollOut_Value
from search.mcts.graph import StateNode
import numpy as np
from search.chess.chess_state import ChessState
from search.chess.pvr_network import policy_nn, rollout_nn
from search.chess.pvr_network import value_nn
import cProfile, pstats, io, time
import argparse
import os

def new_chess_game():
    x=np.zeros([10,9])
    x[0,:]=np.array([4,3,5,6,7,6,5,3,4])
    x[3,:]=np.array([1,0,1,0,1,0,1,0,1])
    x[2,1]=2
    x[2,7]=2
    x=x-np.flipud(x)
    return np.int32(x)

def move_one_step(pos, movei):
    if pos[movei[0],movei[1]] != 0:
        pos[movei[2],movei[3]] = pos[movei[0],movei[1]]
        pos[movei[0],movei[1]] = 0
    else:
        ValueError("There is no chess in (%d,%d) and the input file is wrong", movei[0],movei[1]) 
    return pos

def play_chess(filename):
    
    
    # read file
    move_dir = "./chess_move"
    det = os.path.join(move_dir,filename)
    det_save = os.path.join(move_dir,'final'+filename) 
    if not os.path.isfile(det):
        open(det, 'w').close()
    fi = open(det, 'r')
    fo = open(det_save, 'w')
    
    move_input = fi.read()
    move_input = move_input.strip()
    move_input = move_input.replace(' ','')
    if len(move_input) > 0:
        move_input = move_input + '\n'
    if len(move_input)%5 != 0:
        ValueError('The format of input file is wrong!\n')

    fo.write(move_input)

    # generate move
    move = []
    moveone = []
    print(move_input)

    for i in range(len(move_input)):
        if i%5 == 4:
            if move_input[i] != '\n':
                ValueError('The format of input file is wrong!\n')
            else:
                move.append(moveone)
                moveone = []
        else: 
            moveone.append(int(move_input[i]))
     
    if len(move)%2 == 0:
        player = 1
    else:
        player = -1

    pr = cProfile.Profile()
    pr.enable()

    pos = new_chess_game()
    for i in range(len(move)):
        movei = move[i]
        pos = move_one_step(pos, movei)
    

    # define the method of mct
    mcts = myMCTS(tree_policy=Go(c=1), 
                default_policy=RandomKStepRollOut_Value(20, 0.9),
                backup=monte_carlo)
    
    # generate the network
    policy_fun = policy_nn()
    rollout_fun = rollout_nn()
    value_fun = value_nn() 

    finish = 0
    for _ in range(400):
        print(np.flipud(pos))
        root = StateNode(None, ChessState(pos, player, policy_fun, rollout_fun, value_fun, False ))
        best_action = mcts(root, n=500) 

        print(best_action.action)
        act = best_action.action
        fo.write(str(act[0])+str(act[1])+str(act[2])+str(act[3])+'\n')
        pos = move_one_step(pos, best_action.action)
        
        for _ in range(100):
            oppo_move =  input("please input the player's move(for example: 2,7,2,4):")
            oppo_move = oppo_move.strip()
            oppo_move = oppo_move.replace(' ','')
            oppo_move = oppo_move.replace(',','')
            if oppo_move == 'no':
                finish = 1
                break
            elif len(oppo_move) == 4:
                moveone = [int(oppo_move[0]),int(oppo_move[1]),int(oppo_move[2]),int(oppo_move[3])]
                fo.write(oppo_move+'\n')
                pos = move_one_step(pos, moveone)
                break
        
        if finish == 1:
            break

    pr.disable()
    s = io.StringIO()
    sortby = 'cumulative'
    ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
    ps.print_stats()
    print(s.getvalue())




if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Chinese Chess AI')
    parser.add_argument('-f', action='store', default='test1.txt',dest='file', help='Add chess move file')
    results = parser.parse_args()
    file = results.file
    play_chess(file)
