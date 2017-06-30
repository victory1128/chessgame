#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: YongfengLi
"""

import ChessStep2 as cs
from policy_network import policy_nn


x = cs.new_chess_game()
action_p = policy_nn(x)

for i in range(len(action_p)):
    print(" %f \n", action_p[i])
