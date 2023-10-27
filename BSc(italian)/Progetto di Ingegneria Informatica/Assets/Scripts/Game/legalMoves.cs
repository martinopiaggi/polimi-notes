using System.Collections;
using System.Collections.Generic;

namespace Game
{
    public static class LegalMoves 
    {
        public static ArrayList compute(ArrayList currentTiles, int sumDices, int sumSelectedTiles)
        {
            var tmp = currentTiles;
            var array = new ArrayList();

            if (sumDices == sumSelectedTiles) return array;
            if (tmp.Contains(sumDices - sumSelectedTiles)) array.Add(sumDices - sumSelectedTiles);
            for (var i = 1; i <= (sumDices - i); i++)
            {
                if (!tmp.Contains(i)) continue;
                for (var x = i + 1; x <= (sumDices - i); x++) 
                {
                    if (!tmp.Contains(x)) continue;
                    if (i + x + sumSelectedTiles == sumDices)
                    {
                        if (!array.Contains(x)) array.Add(x);
                        if (!array.Contains(i)) array.Add(i);
                    }
                    else
                    {
                        for (var y = x + 1; y <= (sumDices - x-i); y++)
                        {
                            if (!tmp.Contains(y)) continue;
                            if (i + x + y + sumSelectedTiles == sumDices)
                            {
                                if (!array.Contains(i)) array.Add(i);
                                if (!array.Contains(x)) array.Add(x);
                                if (!array.Contains(y)) array.Add(y);
                            }
                            else
                            {
                                for (var z = y + 1; z <= (sumDices - y-x-i); z++)
                                {
                                    if (!tmp.Contains(z)) continue;
                                    if (i + x + y + z != sumDices) continue; //4 combinations in case of sumSelectedTiles=0; Very rare case. 
                                    if (!array.Contains(i)) array.Add(i);
                                    if (!array.Contains(x)) array.Add(x);
                                    if (!array.Contains(y)) array.Add(y);
                                    if (!array.Contains(z)) array.Add(z);
                                }
                            }
                        
                        }
                    }
                }
            }
            return array;
        }
    
        public static ArrayList computeSets(ArrayList currentTiles, int sumDices)
        {
            if (currentTiles.Count == 0) return new ArrayList();
            
            var tmp = currentTiles;
            var sets = new ArrayList();

            if (tmp.Contains(sumDices))
            {
                HashSet<int> newSet = new HashSet<int>();
                newSet.Add(sumDices);
                sets.Add(newSet);
            }
            for (int i = 1; i <= (sumDices - i); i++)
            {
                if (!tmp.Contains(i)) continue;
                for (var x = i + 1; x <= (sumDices - i); x++)
                {
                    if (!tmp.Contains(x)) continue;
                    if (i + x == sumDices)
                    {
                        HashSet<int> newSet = new HashSet<int>();
                        newSet.Add(i);
                        newSet.Add(x);
                        sets.Add(newSet);
                    }
                    else
                    {
                        for (var y = x + 1; y <= (sumDices - x-i); y++)
                        {
                            if (!tmp.Contains(y)) continue;
                            if (i + x + y == sumDices)
                            {
                                HashSet<int> newSet = new HashSet<int>();
                                newSet.Add(i);
                                newSet.Add(x);
                                newSet.Add(y);
                                sets.Add(newSet);
                            }
                            else
                            {
                                for (var z = y + 1; z <= (sumDices - y-x-i); z++)
                                {
                                    if (!tmp.Contains(z)) continue;
                                    if (i + x + y + z != sumDices) continue; //4 combinations in case of sumSelectedTiles=0; Very rare case. 
                                    HashSet<int> newSet = new HashSet<int>();
                                    newSet.Add(i);
                                    newSet.Add(x);
                                    newSet.Add(y);
                                    newSet.Add(z);
                                    sets.Add(newSet);
                                }
                            }
                        }
                    }
                }
            }
            return sets;
        }
    }
    
}
