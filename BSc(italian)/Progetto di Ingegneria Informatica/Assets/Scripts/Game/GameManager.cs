using System.Collections;
using System.Collections.Generic;
using Game.ui;
using Players;
using UnityEngine;

namespace Game
{
    public class GameManager : MonoBehaviour
    {
        public DiceLauncher launcher;
        public float seatDistance;
        public float seatHeight;
        public GameObject player;
        public GameObject aiPlayer;
        public GameObject menuUI;
        public GameObject inGameUI;
        private string _humanUsername;
        protected int numPlayers;
        [SerializeField]protected int numOfTiles = 12;
        protected int sumValue;
        protected int sumSelectedTiles;
        private List<GameObject> _playersPlaying;
        private List<GameObject> _playersOut;
        private GameObject _currentPlayer;
    

        public void Awake()
        {
            launcher = this.GetComponent<DiceLauncher>();
        }

        public void StartGame()
        {
            _playersPlaying = new List<GameObject>(numPlayers);
            _playersOut = new List<GameObject>();
            //main player
            var posTiles = new Vector3(0, seatHeight, -seatDistance);
            var humanPlayer = Instantiate(player, posTiles, Quaternion.identity);
            humanPlayer.GetComponent<Player>().startPlaying(this, numOfTiles, _humanUsername,false);
            _playersPlaying.Add(humanPlayer);


            //left player
            if (numPlayers == 4)
            {
                posTiles = new Vector3(-seatDistance, seatHeight, 0);
                var leftPlayer = Instantiate(aiPlayer, posTiles, Quaternion.LookRotation(-posTiles));
                leftPlayer.GetComponent<AIPlayer>().startPlaying(this, numOfTiles, "Monte",false);
                _playersPlaying.Add(leftPlayer);
            }

            //front player 
            posTiles = new Vector3(0, seatHeight, seatDistance);
            var frontPlayer = Instantiate(aiPlayer, posTiles, Quaternion.LookRotation(-posTiles));
            frontPlayer.GetComponent<AIPlayer>().startPlaying(this, numOfTiles, "Carlo",false);
            _playersPlaying.Add(frontPlayer);

            //right player 
            if (numPlayers == 4)
            {
                posTiles = new Vector3(seatDistance, seatHeight, 0);
                var rightPlayer = Instantiate(aiPlayer, posTiles, Quaternion.LookRotation(-posTiles));
                rightPlayer.GetComponent<AIPlayer>().startPlaying(this, numOfTiles, "Sir Tree",false);
                _playersPlaying.Add(rightPlayer);
            }
        
            _currentPlayer = _playersPlaying[0];
            this.transform.position = _currentPlayer.transform.position;
            this.transform.rotation = _currentPlayer.transform.rotation;
            StartCoroutine(launcher.turnStart(3f));
        }
    
        public void DicesStopped(int sum)
        {
            sumValue = sum;
            UpdatingPlayerTiles();
        }

        private void UpdatingPlayerTiles()
        {
            ArrayList selectableTiles = LegalMoves.compute(_currentPlayer.GetComponent<Player>().GetTiles(), sumValue, sumSelectedTiles);
            if (selectableTiles.Count > 0)
            {
                _currentPlayer.GetComponent<Player>().EnableSelect(true);
                _currentPlayer.GetComponent<Player>().SetPlayerSelectables(selectableTiles,sumValue-sumSelectedTiles);
            }
            else if (sumSelectedTiles == sumValue) //the player have selected all the tiles necessary to reach the dices sum, he ended his turn
            {
                sumSelectedTiles = 0;
                _currentPlayer.GetComponent<Player>().EnableSelect(false);
                if (_currentPlayer.GetComponent<Player>().GetTiles().Count == 0) //Check in case of immediate win
                {
                    var allPlayers = new List<GameObject>();
                    foreach (var p in _playersPlaying) allPlayers.Add(p);
                    foreach (var p in allPlayers) PlayerGameOver(p); //immediate win, immediate Game Over for everyone
                    
                }
                else
                {
                    if(_playersPlaying.Count>0)ChangePlayer();
                }
            }
            else //the player hasn't enough tiles to reach the dices sum, game over for the player
            {
                var deletedPlayer = _currentPlayer;
                deletedPlayer.GetComponent<Player>().EnableSelect(false);
                if(_playersPlaying.Count-1>0)ChangePlayer();
                PlayerGameOver(deletedPlayer);
            }
        }
    
        private void PlayerGameOver(GameObject eliminatedPlayer)
        {
            int score = 0;
            var tiles = eliminatedPlayer.GetComponent<Player>().GetTiles();
            foreach (int number in tiles) score += number; //counting the score
            score = sum(numOfTiles) - score;
            eliminatedPlayer.GetComponent<Player>().SetScore(score);
            _playersPlaying.Remove(eliminatedPlayer);
            _playersOut.Add(eliminatedPlayer);
            if (_playersPlaying.Count == 0) GameOver();
            inGameUI.GetComponent<UIManager>().UpdateScores(_playersOut); //updates the scores
        }

        private void GameOver()
        {
            GameObject winner = null;
            var winnerScore = int.MinValue;
            foreach (GameObject p in _playersOut)
            {
                var playerScore = p.GetComponent<Player>().GetScore();
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
            if(winner==null)inGameUI.GetComponent<UIManager>().DeclareTie();
            else inGameUI.GetComponent<UIManager>().DeclareWinner(winner.GetComponent<Player>().GetUsername());
        }

        private void ChangePlayer()
        {
            int currentIndex = _playersPlaying.IndexOf(_currentPlayer);
            _currentPlayer = _playersPlaying[(currentIndex + 1)%(_playersPlaying.Count)]; //circular array
            launcher.transform.position = _currentPlayer.transform.position;
            launcher.transform.rotation = _currentPlayer.transform.rotation;
            launcher.enabled = true;
            StartCoroutine(launcher.turnStart(0.5f));
        }

        public void SelectTile(int selectedTile)
        {
            sumSelectedTiles += selectedTile;
            UpdatingPlayerTiles();
        }


        public void setNumOfPlayers(int num)
        {
            this.numPlayers = num;
        }

        public void setNumOfTiles(int num)
        {
            this.numOfTiles = num;
        }

        public void setHumanNickname(string username)
        {
            _humanUsername = username;
        }

        public int getNumOfPlayers()
        {
            return numPlayers;
        }

        public string getHumanNickname()
        {
            return _humanUsername;
        }


        public static int sum(int numOfTiles)
        {
            int sum = 0;
            for (int x = 1; x <= numOfTiles; x++) sum += x;
            return sum;
        }

        public int maximumScore()
        {
            return sum(numOfTiles);
        }

        public void newPlay()
        {
            StartCoroutine(menuUI.GetComponent<MenuManager>().endPlay());
            foreach(GameObject tile in GameObject.FindGameObjectsWithTag("tile")) 
                Destroy(tile);
            foreach(GameObject p in GameObject.FindGameObjectsWithTag("Player")) 
                Destroy(p);
        }
        public void setDicesToDefaultPosition()
        {
            launcher.setDicesToDefaultPosition();
        }
    }
}