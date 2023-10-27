using System.Collections;
using System.Collections.Generic;
using Game;

namespace Players
{
    public class AIPlayer : Player
    {
        private bool _computing;
        private bool _computed;
        private Queue bestMove;

        private void Update() //used during a normal game with human player
        {
            if (!selectEnabled) return; //selectEnabled is true when the Game Manager receive the result of the dices from the DicesLauncher 
            if (_computing) return;
            if (_computed) //when computed playerAI knows which are the best moves and play them 
            {
                int tileToSelect = (int) bestMove.Dequeue();
                tilesObj[tileToSelect-1].GetComponent<Tile>().Flip();
                if (bestMove.Count == 0) _computed = false; //my turn has finished
            }
            else
            {
                Mcts.Mcts oracle = new Mcts.Mcts(gameMan.maximumScore(),this);
                var sumDices = remainingValue;
                _computing = true;
                StartCoroutine(oracle.ComputeBestMove(100000,tiles, sumDices));
            }
        }

        public void takeAdvice(HashSet<int> bestMove) //used to receive the best move from MonteCarlo Tree Search Algorithm
        {
            this.bestMove = new Queue();
            foreach (int tile in bestMove)
            {
                this.bestMove.Enqueue(tile);
            }
            _computing = false;
            _computed = true;
        }
        
        public override int ReturnTileTestBench() //used during the TestBench  
        {
            if (_computing) return 0;
            if (_computed) 
            {
                var tileToSelect = (int) bestMove.Dequeue();
                if (bestMove.Count == 0) _computed = false; //my turn has finished
                tiles.Remove(tileToSelect);
                return tileToSelect;
            }

            Mcts.Mcts oracle = new Mcts.Mcts(gameMan.maximumScore(),this);
            var sumDices = remainingValue;
            _computing = true;
            StartCoroutine(oracle.ComputeBestMove(50000,tiles, sumDices));
            return 0;
        }

    }
}

