
# Chinese Chess AI
I'm a chinese chess AI. I will become stronger and stronger.

# Author
 * Yongfeng Li
 * Ruosi Wang
 * Bing Yu

## main
- please run `python3 chess_main.py -f test.txt` to run our Chinese Chess AI. You can find the results in chess_move directory.

## spider
- the chess data is from http://www.01xq.com/. 
- If you want more, just modify the range in chessspider/spiders/chessdataspider and run `scrapy crawl chessspider -o new.json` in chessspider directory

## train
To train different networks:
- cd train_code
- Run gen_data_new.py to generate the data (may take dozens of minutes to several hours).
- Run SL_predict_emb.py to train the network with attention along lines layers and improved feature map.
- Run SL_predict_new.py to train the network with multi-head attention and unimproved feature map.
- Run SL_predict_emb_simplified.py to train the baseline network
- Run rollout.py to train the rollout network
- Run QNET.py to train the value network (the network similar to the network in SL_predict_emb.py)
- the RL_train.py is for the old version SL network and dataset, we only put it here.

## the directory
 * search: contains all code about Monte Carlo Tree Search and the network interface.
 * chessspider: contains the code for spider
 * param: contains the checkpoint of all networks  	
 * chess_move: constains the input and output for chess_main.py 
 * value: contains all the value network and trainning date
 * train_code: contains the code of train data 


