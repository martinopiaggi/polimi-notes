using System.Collections;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace Game.ui
{
    public class MenuManager : MonoBehaviour
    {
        public GameObject titleObj;
        private TextMeshProUGUI titleGUI;

        public GameObject numOfPlayerObj;
        public GameObject numOfTilesObj;

        public GameObject nicknameInput;

        private TMP_InputField  nicknameInputField;

        public GameObject gameMan;
        public GameObject inGameCanvas;
        private GameManager game;
        void Start()
        {
            QualitySettings.vSyncCount = 0;
            Application.targetFrameRate = 60;
            game = gameMan.GetComponent<GameManager>();
            titleGUI = titleObj.GetComponent<TextMeshProUGUI>();
            nicknameInputField = nicknameInput.GetComponent<TMP_InputField>();
            numOfPlayerObj.SetActive(false);
            numOfTilesObj.SetActive(false);
            nicknameInput.SetActive(false);
            inGameCanvas.SetActive(false);
            titleGUI.alpha = 1f;
            StartCoroutine(mainTitle());        
        }

        private IEnumerator mainTitle(){
            yield return new WaitForSeconds(1f);
            yield return StartCoroutine(fadeOut(this.GetComponent<CanvasGroup>()));
            StartCoroutine(inputNicknameTitle());
        }

        private IEnumerator inputNicknameTitle(){
            titleObj.SetActive(false);
            nicknameInput.SetActive(true); //activation of Input Field for Nickname
            yield return StartCoroutine(fadeIn(this.GetComponent<CanvasGroup>()));
            nicknameInputField.Select(); //directly select the Input Field
        }

        private IEnumerator fadeOut(CanvasGroup canvas){
            var opacity = 1f;
            while(opacity>=0f){
                opacity-=0.033f; 
                canvas.alpha=opacity; 
                yield return new WaitForSeconds(0.01f);
            }
        }

        private IEnumerator fadeIn(CanvasGroup canvas){
            var opacity = 0f;
            while(opacity<=1f){
                opacity+=0.033f;
                canvas.alpha=opacity; 
                yield return new WaitForSeconds(0.01f);
            }
        }

        public void inputFieldDeselected(){
            string nickname = nicknameInputField.text;
            if(!nickname.Equals("Insert nickname")&&!nickname.Equals("")){
                game.setHumanNickname(nickname);
                StartCoroutine(numOfPlayersTitle());
            }
        }

        private IEnumerator numOfPlayersTitle(){
            yield return StartCoroutine(fadeOut(this.GetComponent<CanvasGroup>())); 
            nicknameInput.SetActive(false);
            numOfPlayerObj.SetActive(true);
            //it's necessary to re-activate buttons since after a click I always disable them
            //after the first match, during the second or more match.. the buttons need to be enabled
            foreach (GameObject b in GameObject.FindGameObjectsWithTag("button")) b.GetComponent<Button>().enabled = true;
            yield return StartCoroutine(fadeIn(this.GetComponent<CanvasGroup>()));
        }

        public void set2PlayersButtonAction(){
            game.setNumOfPlayers(2);
            StartCoroutine(numOfTiles());
        }

        public void set4PlayersButtonAction(){
            game.setNumOfPlayers(4);
            StartCoroutine(numOfTiles()); 
        }

        public IEnumerator numOfTiles(){
            yield return StartCoroutine(fadeOut(this.GetComponent<CanvasGroup>()));
            numOfPlayerObj.SetActive(false);
            numOfTilesObj.SetActive(true); 
            //it's necessary to re-activate buttons since after a click I always disable them
            //after the first match, during the second or more match.. the buttons need to be enabled
            foreach (GameObject b in GameObject.FindGameObjectsWithTag("button")) b.GetComponent<Button>().enabled = true; 
            StartCoroutine(fadeIn(this.GetComponent<CanvasGroup>()));
        }

        public void set9TilesButtonAction(){
            game.setNumOfTiles(9);
            StartCoroutine(startGame());
        }

        public void set12TilesButtonAction(){
            game.setNumOfTiles(12);
            StartCoroutine(startGame());
        }


        private IEnumerator startGame(){
            yield return StartCoroutine(fadeOut(this.GetComponent<CanvasGroup>()));
            numOfTilesObj.SetActive(false);
            game.StartGame();
            yield return new WaitForSeconds(3f);
            var inGameUIcanvas = inGameCanvas.GetComponent<CanvasGroup>();
            inGameCanvas.SetActive(true);
            inGameCanvas.GetComponent<UIManager>().StartMatch();
            StartCoroutine(fadeIn(inGameUIcanvas));
        }

        public IEnumerator endPlay()
        {
            var inGameUIcanvas = inGameCanvas.GetComponent<CanvasGroup>();
            yield return StartCoroutine(fadeOut(inGameUIcanvas));
            inGameCanvas.GetComponent<UIManager>().ResetScore();
            inGameCanvas.SetActive(false);
            StartCoroutine(numOfPlayersTitle());
        }
    }
}
