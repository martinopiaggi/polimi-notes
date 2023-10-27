using UnityEngine;

namespace Game
{
    public class Dice : MonoBehaviour
    {
        private Transform diceTransf;
        private bool isJustLaunched;
        private Rigidbody thisRB;
        private int value;

        public void Start(){ 
            thisRB = this.GetComponent<Rigidbody>();
            ResetValue();
        }

        public int GetValue()
        { return value; }
    
        public void ResetValue(){value = -1;}

        public void Launched()
        {
            isJustLaunched = true;
        }
    
        private void OnCollisionExit(Collision collision)
        {
            if (isJustLaunched)
            {
                if (thisRB.angularVelocity.magnitude <=0.001f)
                {
                    UpdateValue();
                }
            }
        } 
    
        private void UpdateValue() //Computing the value of the dice using dot product of the orientation vectors 
        {
            diceTransf = this.GetComponent<Transform>();
            if (Vector3.Dot(diceTransf.up, Vector3.up) > 0.9f) value = 3;
            else if (Vector3.Dot(-diceTransf.up, Vector3.up) > 0.9f) value = 4;
            else if  (Vector3.Dot(diceTransf.forward, Vector3.up) > 0.9f) value = 1;
            else if (Vector3.Dot(-diceTransf.forward, Vector3.up) > 0.9f) value = 6;
            else if (Vector3.Dot(diceTransf.right, Vector3.up) > 0.9f) value = 2;
            else if (Vector3.Dot(-diceTransf.right, Vector3.up) > 0.9f) value = 5;
            else ResetValue();
            isJustLaunched = false;
            thisRB.isKinematic = true;
        }
    
  
    }
}
