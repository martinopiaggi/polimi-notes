using System.Collections;
using UnityEngine;

namespace Players
{
    public class RandomPlayer : Player
    {
        private bool _computing;
        private bool _computed;
        private Queue bestMove;

        public override int ReturnTileTestBench()
        {
            //selectable tiles are set by TestBench
            var selected = (int)selectableTiles[Random.Range(0, selectableTiles.Count)]; //random selection from selectable tiles
            tiles.Remove(selected);
            return selected; 
        }
    
    
    }
}