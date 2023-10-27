using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class PlayerStats 
{
    private int totalMatches;
    private int totalWins;
    private float averageScore;


    public PlayerStats()
    {
        totalMatches = 0;
        totalWins = 0;
        averageScore = 0f;
    }

    public void win()
    {
        totalWins++;
    }
    
    public float getRatioWins()
    {
        return (float)totalWins / totalMatches;
    }

    public void newScore(int newScore)
    {
        if (totalMatches == 0)
        {
            averageScore = (float)newScore;
            totalMatches++;
            return;
        }
        var newAverageScore = averageScore * totalMatches + newScore;
        totalMatches++;
        averageScore = newAverageScore / totalMatches;
    }

    public int getAverageScore()
    {
        return (int)averageScore;
    }
    
}
