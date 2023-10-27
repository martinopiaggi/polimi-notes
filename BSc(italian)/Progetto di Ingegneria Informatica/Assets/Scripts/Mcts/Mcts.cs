using System;
using System.Collections;
using System.Collections.Generic;
using Game;
using Players;
using UnityEngine;
using Random = UnityEngine.Random;

namespace Mcts
{
    public class Mcts
    {
        private readonly int _maximumScore;
        private readonly AIPlayer _caller;
        private ArrayList _tilesRoot;
        private State root;
        
        public Mcts(int maximumScore, AIPlayer caller)
        {
            _maximumScore = maximumScore;
            _caller = caller;
        }


        public IEnumerator ComputeBestMove(int mctsIterations, ArrayList tilesRoot, int sumDices){
            root = new State(null, tilesRoot, null);
            for(int i = 2; i<=12;i++)root.expandPlay(i); //making the root already expanded, no necessity to explore different launches at root node 

            //generate children root
            var possibleMoves = LegalMoves.computeSets(tilesRoot, sumDices);
            var bestMove = (HashSet<int>)possibleMoves[0]; 
        
            //compute best move only when there is at least one move
            if (possibleMoves.Count > 1){
                foreach (HashSet<int> possibleMove in possibleMoves)
                {
                    var newChild = new State(root, TilesAfterPlay(tilesRoot,possibleMove), possibleMove);
                    root.addChild(newChild);
                }

                while (root.getSimulations() < mctsIterations)
                {
                    var selected = Selection(root);
                    var expanded = Expansion(selected);
                    var simulated = Simulation(expanded);
                    Backup(simulated);
                    if (root.getSimulations() % 1000 == 0)
                    {
                        yield return null;
                    } 
                }
                bestMove = FindBestWinRateFromChildren(root).getPlayed();
            }
            _caller.takeAdvice(bestMove);
        }

        private State Selection(State state)
        {
            var selected = state;
            while (selected.isFullExpanded()&&selected.getChildren().Count>0) selected = FindBestUcbFromChildren(selected);
            return selected;
        }
        private State Expansion(State state)
        {
            if (state.isFullExpanded()) return state;
        
            var randomPlay = (int)state.getUnexpandedPlays()[Random.Range(0, state.getUnexpandedPlays().Count)];
            state.expandPlay(randomPlay);
            var possibleMoves = LegalMoves.computeSets(state.getTiles(), randomPlay);
            if (possibleMoves.Count == 0) return state;
            var childrenToSimulate = new ArrayList();
            foreach (HashSet<int> possibleMove in possibleMoves)
            {
                var newChild = new State(state, TilesAfterPlay(state.getTiles(), possibleMove), possibleMove);
                state.addChild(newChild);
                childrenToSimulate.Add(newChild);
            }
            state = (State)childrenToSimulate[Random.Range(0, childrenToSimulate.Count)];
            return state;
        }
        private State Simulation(State state)
        {
            var sumDices = RandomLaunch();
            while (true)
            {
                var possibleMoves = LegalMoves.computeSets(state.getTiles(), sumDices);
                if (possibleMoves.Count > 0)
                {
                    var randomMove = (HashSet<int>)possibleMoves[Random.Range(0, possibleMoves.Count)];
                    var tilesAfterPlay = this.TilesAfterPlay(state.getTiles(), randomMove);
                    state = new State(state, tilesAfterPlay, randomMove);
                }
                else return state;
            }
        }
        private void Backup(State state)
        {
            var discount = 0.7f;
            var newScore = ComputeScore(state);
            while (state.getParent() != null) 
            {
                state.addScore((int)(newScore*discount));
                state.increaseSimulations();
                discount *= discount;
                state = state.getParent();
            }
            state.increaseSimulations(); //root 
        }
        private int ComputeScore(State leaf)
        {
            var score = _maximumScore;
            foreach (int tile in leaf.getTiles()) score -= tile;
            return score;
        }
        private int RandomLaunch()
        {
            var dicesValue = Random.Range(1, 7);
            dicesValue+=Random.Range(1, 7);
            return dicesValue;
        }
        private State FindBestUcbFromChildren(State state)
        {
            var maxUcb = float.MinValue;
            var bestOne = state;
            foreach (State child in state.getChildren())
            {
                var newUcb = ComputeUcb(child);
                if (newUcb < maxUcb) continue;
                maxUcb = newUcb;
                bestOne = child;
            }
            return bestOne;
        }
        private State FindBestWinRateFromChildren(State state)
        {
            var maxWinRate = float.MinValue;
            var bestOne = state;
            foreach (State child in state.getChildren())
            {
                var newWinRate = ComputeWinRate(child);
                if (newWinRate < maxWinRate) continue;
                maxWinRate = newWinRate;
                bestOne = child;
            }
            return bestOne;
        }
        private float ComputeUcb(State state)
        {
            if (state.getSimulations() == 0) return float.MaxValue;
            if(state.getParent().getSimulations()==0) return float.MaxValue;
            var newWinRate = ComputeWinRate(state);
            var newUcb = newWinRate + 1.41f * (float)Math.Sqrt(Math.Log(state.getParent().getSimulations()) / state.getSimulations());
            state.ucb = newUcb;
            return newUcb;
        }
        private float ComputeWinRate(State state)
        {
            if(state.getParent().getSimulations()==0) return 0f;
            var newWinRate = (float)state.getHeritageScore() / (_maximumScore * state.getSimulations());
            state.winRate = newWinRate;
            return newWinRate;
        }
        private ArrayList TilesAfterPlay(ArrayList tiles, HashSet<int> move)
        {
            var tilesAfterMove = new ArrayList();
            foreach (var tile in tiles) if (!move.Contains((int)tile)) tilesAfterMove.Add(tile);
            return tilesAfterMove;
        }
    }
}