# A* algorithm visualization in Java 
![astar1](https://user-images.githubusercontent.com/72280379/155737789-a7460193-9511-4a0a-8652-6f3bb6572d49.gif)

A* search algorithm visualization written in Java with Processing graphic library.

#### Functions overview 
```Java
public void settings()
public void setup() //Processing function, the first that is called
public void setupGrid() //makes the grid
public void drawGrid() //draws the grid
public void mousePressed() //Processing function, used to determine when mouse is pressed
public void aStar() //the search algorithm
public float heuristic() //the Heuristic function used by the algorithm
public void draw() //Processing function used to update the screen
public void drawPath() //it draws the path starting from the Target node and backtracks all the predecessors
```
#### Graphics
```Java
public void settings() {  
    w = 800;  //width in pixels of window
    h = 800;  //height of window in pixels
    sizeNode = 15; //always pixels 
    bgColor = new Color(0x03045e);  
    obstacleColor = new Color(0x0077b6);  
    lineColor = new Color(0xcaf0f8);  
    size(w,h);  
}
```
You can easily change the dimensions of the window, the size of the nodes (smaller nodes = slower program) and the colors. 


#### You need the Processing library 
[Just download from here](https://processing.org/download) and remember to import it in your IDE. For example in Intellij Idea, select and add the folder 'processing/core/library' in Project Structure/Project Settings/Libraries from the editor. 

