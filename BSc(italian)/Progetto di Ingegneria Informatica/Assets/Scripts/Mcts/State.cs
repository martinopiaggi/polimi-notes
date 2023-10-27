using System.Collections;
using System.Collections.Generic;

namespace Mcts
{
    public class State
    {
        private State parent;
        private ArrayList tiles;
        private HashSet<int> move;
        private ArrayList children;
        public float ucb;
        public float winRate;
        private int heritageScore;
        private int simulations;
        private ArrayList unexpandedPlays;
    
    
        public State(State parent, ArrayList tiles, HashSet<int> move)
        {
            this.parent = parent;
            this.tiles = tiles;
            this.move = move;
            heritageScore = 0;
            simulations = 0;
            children = new ArrayList();
            unexpandedPlays = new ArrayList();
            for (var i = 2; i <= 12; i++) unexpandedPlays.Add(i);
        }

        public State getParent()
        {
            return parent;
        }

        public ArrayList getTiles()
        {
            return tiles;
        }

        public ArrayList getChildren()
        {
            return children;
        }

        public int getHeritageScore()
        {
            return heritageScore;
        }

        public int getSimulations()
        {
            return simulations;
        }

        public ArrayList getUnexpandedPlays()
        {
            return unexpandedPlays;
        }

        public bool isFullExpanded()
        {
            if (tiles.Count == 0) return true;
            return unexpandedPlays.Count == 0;
        }

        public void expandPlay(int dicePlay)
        {
            unexpandedPlays.Remove(dicePlay);
        }

        public void addChild(State state)
        {
            children.Add(state);
        }

        public void addScore(int newScore)
        {
            heritageScore += newScore;
        }

        public void increaseSimulations()
        {
            simulations++;
        }

        public HashSet<int> getPlayed()
        {
            return move;
        }

    }
}
