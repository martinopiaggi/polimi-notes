# Computer Science Project - MCTS

A little game where you play against 1-3 AIplayers. 
Made in the Unity game engine, the focus of this project is the implementation of a simple board game called 'Shut The Box' and the heuristic search algorithm Monte Carlo Tree Search.
MCTS algorithm is used to solve the game tree and permits to the AI to play the best move possible.

![shutTheBoxGIF](https://user-images.githubusercontent.com/72280379/183460879-b14aa260-955e-4dff-917d-7bba9a5dadaf.gif)

# Shut The Box rules
Here the wikipedia page of Shut The Box rules:
https://en.wikipedia.org/wiki/Shut_the_box

# Useful docs about MCTS 

Useful introduction to MCTS algorithm: https://towardsdatascience.com/monte-carlo-tree-search-an-introduction-503d8c04e168

Wikipedia MCTS page: https://en.wikipedia.org/wiki/Monte_Carlo_tree_search

Good paper about MCTS methods: 
https://www.researchgate.net/publication/235985858_A_Survey_of_Monte_Carlo_Tree_Search_Methods

# Testbench

*Sum of percentages != 100% because a match could end in a draw*

### AI vs RandomPlayer (100 matches, 9 tiles, MCTS 20k iterations)

|      Agent       |  Wins  | Average score |  
|:-----:|:-----:|:-----:|
| RandomPlayer | 31% | 23/45 | 
|        AI        | 62% | 28/45 |  


### AI vs RandomPlayer (200 matches, 12 tiles, MCTS 50k iterations)

|      Agent       |  Wins  | Average score |  
|:-----:|:-----:|:-----:|
| RandomPlayer  | 20%| 25/78  | 
| AI | 77% | 36/78 | 


### AI vs RandomPlayer (20 matches, 12 tiles,MCTS 500k iterations)

|      Agent       |  Wins  | Average score | 
|:-----:|:-----:|:-----:|
| RandomPlayer  | 11%| 23/78  | 
| AI | 88% | 42/78 |   

Introduction of a Strategy Player: a player who always plays the combination which involves as few tiles as possible. 

### Strategy vs Random (1000 matches, 12 tiles)

|      Agent       |  Wins  | Average score | 
|:-----:|:-----:|:-----:|
| RandomPlayer  | 14%| 25/78  | 
| StrategyPlayer | 83% | 42/78 |   

### Strategy vs AI (100 matches, 12 tiles, MCTS 100k iterations)

|     Agent      |  Wins  | Average score | 
|:--------------:|:-----:|:-----:|
|       AI       | 42% | 38/78 |  
| StrategyPlayer | 57% | 42/78 |  

### Strategy vs AI (50 matches, 12 tiles, MCTS 1 million iterations)

|      Agent       |  Wins  | Average score | 
|:-----:|:-----:|:-----:|
| AI | 47% | 41/78 |  
| StrategyPlayer | 44% | 41/78 |  

