using System.Collections;

namespace Players
{
    public class StrategyPlayer : Player
    {
        private bool _computing;
        private bool _computed;
        private Queue bestMove;


        public override int ReturnTileTestBench()
        {
            //selectable tiles are set by TestBench
            var selected = (int)selectableTiles[0]; //selecting first tile of selectables gives always the move with less tiles (the simplest combination)
            tiles.Remove(selected);
            return selected; 
        }
    
    
    
    
    
    
    }
}