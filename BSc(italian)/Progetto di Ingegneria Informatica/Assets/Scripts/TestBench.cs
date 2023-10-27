using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Game;
using Players;
using UnityEngine;
using UnityEngine.Serialization;
using static System.String;
using Random = UnityEngine.Random;


public class TestBench : GameManager
{
    private List<Player> _playing;
    private List<Player> _out;
    private Player _current;
    public int numberOfMatches = 1000;
    public int numOfRandomPlayers = 4;
    public int numOfAi = 4;
    public int numOfStrategyPlayers = 4;
    
    public void Start()
    {
        numPlayers = numOfRandomPlayers + numOfAi + numOfStrategyPlayers; 
        
        _playing = new List<Player>(numPlayers);
        _out = new List<Player>();

        for (var i = 0; i < numOfRandomPlayers; i++)
        {
            Player p = null;
            GameObject newPlayer = new GameObject("Random");
            p = newPlayer.AddComponent<RandomPlayer>();
            _playing.Add(p);
            p.KeepStatistics();
            p.startPlaying(this, numOfTiles, "testbench",true);
        }
        
        for (var i = 0; i < numOfAi; i++)
        {
            Player p = null;
            GameObject newPlayer = new GameObject("AI");
            p = newPlayer.AddComponent<AIPlayer>();
            _playing.Add(p);  
            p.KeepStatistics();
            p.startPlaying(this, numOfTiles, "testbench",true);
        }
        
        for (var i = 0; i < numOfStrategyPlayers; i++)
        {
            Player p = null;
            GameObject newPlayer = new GameObject("StrategyPlayer");
            p = newPlayer.AddComponent<StrategyPlayer>();
            _playing.Add(p);  
            p.KeepStatistics();
            p.startPlaying(this, numOfTiles, "testbench",true);
        }
        
        _out = new List<Player>();
        StartCoroutine(Bench());
    }

    private IEnumerator Bench()
    {
        sumValue = Random.Range(1, 7) + Random.Range(1, 7);
        sumSelectedTiles = 0;
        var matches = 1;
        while (matches <= numberOfMatches)
        {
            _current = _playing[Random.Range(0, _playing.Count)]; //random init player of the match
            while (_playing.Count>0)
            {
                sumValue = Random.Range(1, 7) + Random.Range(1, 7); //random launch
                var selectableTiles = UpdatingPlayerTiles();
                while (selectableTiles.Count>0) 
                {
                    var selected = 0;
                    while (selected == 0)
                    {
                        if(_current.GetType() == typeof(AIPlayer))yield return new WaitForSeconds(0.01f); //waiting MCTS computing 
                        selected = _current.ReturnTileTestBench();
                    }
                    sumSelectedTiles += selected;
                    selectableTiles = UpdatingPlayerTiles();
                }
                TurnFinished();
            }
            
            Debug.Log("FINISHED MATCH, number of matches: "+ matches);
            foreach (var p in _out)
            {
                Debug.Log(p.GetType() + " won "+ (int)(p.ReturnStats().getRatioWins()*100) +" % of total matches");
                Debug.Log(" with average score of "+ p.ReturnStats().getAverageScore());
            }

            matches++;
            _playing = _out;
            foreach (var p in _playing) p.startPlaying(this,numOfTiles,"test",true);
            
            _out = new List<Player>();
        }
    }
    
    private ArrayList UpdatingPlayerTiles()
    {
        ArrayList selectableTiles = LegalMoves.compute(_current.GetComponent<Player>().GetTiles(), sumValue, sumSelectedTiles);
        if (selectableTiles.Count > 0)
        {
            _current.GetComponent<Player>().SetPlayerSelectables(selectableTiles,sumValue-sumSelectedTiles);
        }
        return selectableTiles;
    }

    private void TurnFinished()
    {
        if (sumSelectedTiles == sumValue) 
        {
            if (_current.GetComponent<Player>().GetTiles().Count == 0) //Check in case of immediate win
            {
                var allPlayers = new List<Player>();
                
                foreach (var p in _playing) allPlayers.Add(p);
                
                foreach (var p in allPlayers) PlayerGameOver(p); //immediate win, immediate Game Over for everyone
            }
            else
            {
                if(_playing.Count>0)ChangePlayer();
            }
        }
        else //the player hasn't enough tiles to reach the dices sum, game over for the player
        {
            var deletedPlayer = _current;
            if (_playing.Count - 1 > 0) ChangePlayer();
            PlayerGameOver(deletedPlayer);
        }
        sumSelectedTiles = 0; //new launch
    }
    
    private void PlayerGameOver(Player eliminatedPlayer)
    {
        var score = 0;
        var tiles = eliminatedPlayer.GetTiles();
        foreach (int number in tiles) score += number; //counting the score
        score = sum(numOfTiles) - score;
        eliminatedPlayer.SetScore(score);
        eliminatedPlayer.UpdateScoreStatistics(score);
        _playing.Remove(eliminatedPlayer);
        _out.Add(eliminatedPlayer);
        Debug.Log(eliminatedPlayer.GetType() + " game over with score: " + eliminatedPlayer.GetScore());
        if (_playing.Count == 0) GameOver();
    }

    private void GameOver()
    {
        Player winner = _out[0];
        var winnerScore = int.MinValue;
        foreach (Player p in _out)
        {
            var playerScore = p.GetScore();
            if (playerScore > winnerScore)
            {
                winner = p;
                winnerScore = playerScore;
            }
            else if (playerScore == winnerScore)
            {
                winner = null;
            }
        }
        
        if(winner!=null) winner.IncreaseWins();
    }

    private void ChangePlayer()
    {
        var currentIndex = _playing.IndexOf(_current);
        _current = _playing[(currentIndex + 1)%(_playing.Count)]; //circular array
    }

}