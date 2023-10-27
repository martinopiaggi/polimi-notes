using System.Collections;
using System.Collections.Generic;
using Players;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Game.ui
{
    public class UIManager : MonoBehaviour
    {
        public GameObject leftPlayerName;
        public GameObject leftPlayerScore;
        public GameObject rightPlayerName;
        public GameObject rightPlayerScore;
        public GameObject humanPlayerName;
        public GameObject humanPlayerScore;
        public GameObject upPlayerName;
        public GameObject upPlayerScore;
        public GameObject winnerTitle;
        public GameObject newPlayButton;
        public GameObject tieTitle;
        public GameObject gameManObj;
        private ArrayList names;
        private ArrayList scores; 

        public void Start(){
            newPlayButton.SetActive(false);
            tieTitle.SetActive(false);
            winnerTitle.SetActive(false);
            names = new ArrayList();
            scores = new ArrayList();
            names.Add(leftPlayerName);
            names.Add(rightPlayerName);
            names.Add(upPlayerName);
            names.Add(humanPlayerName);
            scores.Add(leftPlayerScore);
            scores.Add(rightPlayerScore);
            scores.Add(upPlayerScore);
            scores.Add(humanPlayerScore);
        }

        private void ScoreHider(){
            foreach(GameObject score in scores){
                var text = score.GetComponent<TextMeshProUGUI>().text;
                var canvas = score.GetComponent<CanvasGroup>();
                if (text.Equals("99")) canvas.alpha = 0f;
                else if(canvas.alpha!=1f)StartCoroutine(FadeIn(canvas));
            }
        }

        public void ResetScore()
        {
            foreach(GameObject score in scores)
            {
                score.GetComponent<TextMeshProUGUI>().text = "99";
            }
        }

        public void StartMatch(){
            var gameMan = gameManObj.GetComponent<GameManager>();
            humanPlayerName.GetComponent<TextMeshProUGUI>().text=gameMan.getHumanNickname();
            if(gameMan.getNumOfPlayers()==2)
            {
                leftPlayerName.SetActive(false);
                rightPlayerName.SetActive(false);
            }
            else
            {
                leftPlayerName.SetActive(true);
                rightPlayerName.SetActive(true);
            }
            newPlayButton.SetActive(false);
            tieTitle.SetActive(false);
            winnerTitle.SetActive(false);
            winnerTitle.GetComponent<TextMeshProUGUI>().text = "The winner is \n";
            ScoreHider();
        }

        public void UpdateScores(List<GameObject> playersOut){
            foreach(GameObject player in playersOut){
                Player playerOut = player.GetComponent<Player>();
                if (playerOut.GetUsername().Equals("Monte")){
                    leftPlayerScore.GetComponent<TextMeshProUGUI>().text=playerOut.GetScore().ToString();
                }
                else if(playerOut.GetUsername().Equals("Carlo")){
                    upPlayerScore.GetComponent<TextMeshProUGUI>().text=playerOut.GetScore().ToString();
                }
                else if(playerOut.GetUsername().Equals("Sir Tree")){
                    rightPlayerScore.GetComponent<TextMeshProUGUI>().text=playerOut.GetScore().ToString();
                }
                else{
                    humanPlayerScore.GetComponent<TextMeshProUGUI>().text=playerOut.GetScore().ToString();
                }
            }
            ScoreHider();
        }

        public void DeclareWinner(string winnerUsername)
        {
            winnerTitle.SetActive(true);
            winnerTitle.GetComponent<TextMeshProUGUI>().text = winnerTitle.GetComponent<TextMeshProUGUI>().text + winnerUsername + "!";
            StartCoroutine(FadeIn(winnerTitle.GetComponent<CanvasGroup>()));
            gameManObj.GetComponent<GameManager>().setDicesToDefaultPosition();
            StartCoroutine(ShowNewPlay());
        }
    
        public void DeclareTie()
        {
            tieTitle.SetActive(true);
            StartCoroutine(FadeIn(tieTitle.GetComponent<CanvasGroup>()));
            gameManObj.GetComponent<GameManager>().setDicesToDefaultPosition();
            StartCoroutine(ShowNewPlay());
        }

        public IEnumerator ShowNewPlay()
        {
            yield return new WaitForSeconds(1f);
            newPlayButton.SetActive(true);
            newPlayButton.GetComponent<Button>().enabled = true;
            StartCoroutine(FadeIn(newPlayButton.GetComponent<CanvasGroup>()));
        }

        private IEnumerator FadeIn(CanvasGroup canvas){
            var opacity = canvas.alpha; 
            while(opacity<=1f){
                opacity+=0.033f;
                canvas.alpha=opacity; 
                yield return new WaitForSeconds(0.01f);
            }
        }
    
        private IEnumerator FadeOut(CanvasGroup canvas)
        {
            var opacity = canvas.alpha;
            while(opacity>=0f){
                opacity-=0.033f; 
                canvas.alpha=opacity; 
                yield return new WaitForSeconds(0.01f);
            }
        }
    }
}
