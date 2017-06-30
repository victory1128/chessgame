from mcts.mcts import MCTS
from mcts.tree_policies import UCB1, Go
from mcts.default_policies import immediate_reward
from mcts.backups import monte_carlo
from mcts.default_policies import RandomKStepRollOut, RandomKStepRollOut_Value
from mcts.graph import StateNode
import numpy as np
from chess.chess_state import ChessState
from chess.pvr_network import policy_nn, rollout_nn
from chess.v_network import value_nn
import cProfile, pstats, io, time





def new_chess_game():
    x=np.zeros([10,9])
    x[0,:]=np.array([4,3,5,6,7,6,5,3,4])
    x[3,:]=np.array([1,0,1,0,1,0,1,0,1])
    x[2,1]=2
    x[2,7]=2
    x=x-np.flipud(x)
    return np.int32(x)
move = [[2,7,2,4],
	[9,7,7,6]]


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


