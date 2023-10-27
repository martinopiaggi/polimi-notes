import processing.core.PApplet;

import java.awt.*;
import java.util.*;

public class AStarAlgorithmVisualization extends PApplet {
    public static void main(String[] args) {
        PApplet.main("AStarAlgorithmVisualization",args);
    }
    int w;
    int h;
    int sizeNode;
    Node[][] grid;
    Node target;
    Node start;
    Node current;
    PriorityQueue<Node> openSet;
    boolean finished;
    Color bgColor;
    Color obstacleColor;
    Color lineColor;

    public void settings() {
        w = 800;
        h = 800;
        sizeNode = 15;
        bgColor = new Color(0xFF08001A, true);
        obstacleColor = new Color(0x7171FF);
        lineColor = new Color(0xFFD500);
        size(w,h);
    }

    public void setup(){
        start=null;
        target=null;
        finished = false;
        setupGrid();
        drawGrid();
    }

    public void setupGrid(){
        grid = new Node[w/ sizeNode][h/ sizeNode];
        for(int x = 0; x<w/ sizeNode; x++){
            for(int y = 0; y<h/ sizeNode; y++){
                grid[x][y] = new Node(x,y,true, MAX_INT);
            }
        }
    }

    public void drawGrid(){
        background(bgColor.getRGB());
        for (int x = 0; x < w / sizeNode; x++) {
            for (int y = 0; y < h / sizeNode; y++) {
                noStroke();
                if (random(0, 1) > 0.51f) grid[x][y].free = false;
                if (grid[x][y].free) fill(bgColor.getRGB());
                else fill(obstacleColor.getRGB());
                circle(grid[x][y].x * sizeNode + sizeNode / 2, grid[x][y].y * sizeNode + sizeNode / 2, sizeNode);
            }
        }
    }

    //mouse event function used to select start and target node and to refresh grid
    public void mousePressed(){
        if ((start==null||target==null)&&(grid[mouseX / sizeNode][mouseY / sizeNode].free)){ //second condition to not select obstacle node
            fill(lineColor.getRGB());
            circle((mouseX/ sizeNode)* sizeNode + sizeNode /2, mouseY/ sizeNode * sizeNode + sizeNode /2, sizeNode);
            if (start == null) {
                start = grid[mouseX / sizeNode][mouseY / sizeNode];
                start.cost = 0;
                println("registered start");
            }
            else if (target == null){
                if (!grid[mouseX / sizeNode][mouseY / sizeNode].equals(start)){
                    target = grid[mouseX / sizeNode][mouseY / sizeNode];
                    println("registered target");
                    aStar(start,target); //start the pathfinder
                }
            }
        }
        else if((start!=null)&&target!=null){ //refresh grid
            setup();
        }
    }

    public void aStar(Node s, Node t){
        openSet = new PriorityQueue<Node>();
        start.cost = 0 + heuristic(start, target);
        openSet.add(start);
        Node expanded;
        while(!openSet.isEmpty()){
                current = openSet.poll();
                if (current.equals(t)){
                    finished = true;
                    current.visited=true;
                    break;
                }
                if(!current.visited){
                    if (current.x > 0) {
                        expanded = grid[current.x - 1][current.y];
                        if(expanded.free && !finished){
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 10;
                            if (expanded.cost > tentativecost){
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x - 1][current.y])) openSet.remove(grid[current.x - 1][current.y]);
                                openSet.add(grid[current.x - 1][current.y]);
                            }
                        }
                    }
                    if (current.x + 1 < w / sizeNode && !finished) {
                        expanded = grid[current.x + 1][current.y];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 10;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x + 1][current.y])) openSet.remove(grid[current.x + 1][current.y]);
                                openSet.add(grid[current.x + 1][current.y]);
                            }
                        }
                    }
                    if (current.y + 1 < h / sizeNode && !finished){
                        expanded = grid[current.x][current.y + 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 10;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x][current.y + 1])) openSet.remove(grid[current.x][current.y + 1]);
                                openSet.add(grid[current.x][current.y + 1]);
                            }
                        }
                    }
                    if (current.y > 0 && !finished) {
                        expanded = grid[current.x][current.y - 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 10;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x][current.y - 1])) openSet.remove(grid[current.x][current.y - 1]);
                                openSet.add(grid[current.x][current.y - 1]);
                            }
                        }
                    }
                    if (current.y > 0 & current.x + 1 < w / sizeNode && !finished) {
                        expanded = grid[current.x + 1][current.y - 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 14;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x + 1][current.y - 1])) openSet.remove(grid[current.x + 1][current.y - 1]);
                                openSet.add(grid[current.x + 1][current.y - 1]);
                            }
                        }
                    }
                    if (current.y > 0 & current.x > 0 && !finished) {
                        expanded = grid[current.x - 1][current.y - 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 14;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x - 1][current.y - 1])) openSet.remove(grid[current.x - 1][current.y - 1]);
                                openSet.add(grid[current.x - 1][current.y - 1]);
                            }
                        }
                    }
                    if (current.y + 1 < h / sizeNode & current.x > 0 && !finished) {
                        expanded = grid[current.x - 1][current.y + 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 14;
                            if (expanded.cost > tentativecost) {
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x - 1][current.y + 1])) openSet.remove(grid[current.x - 1][current.y + 1]);
                                openSet.add(grid[current.x - 1][current.y + 1]);
                            }
                        }
                    }
                    if (current.y + 1 < h / sizeNode & current.x + 1 < w / sizeNode && !finished) {
                        expanded = grid[current.x + 1][current.y + 1];
                        if (expanded.free && !finished) {
                            int tentativecost = current.cost - heuristic(current, t) + heuristic(expanded, t) + 14;
                            if (expanded.cost > tentativecost){
                                expanded.cost = tentativecost;
                                expanded.predecessor = current;
                                if(openSet.contains(grid[current.x + 1][current.y + 1])) openSet.remove(grid[current.x + 1][current.y + 1]);
                                openSet.add(grid[current.x + 1][current.y + 1]);
                            }
                        }
                    }
                    current.visited = true;
                }
        }
        if(!finished){
                println("Path doesn't exist.");
        }
    }

    public int heuristic(Node a,Node b){
        //Octile distance, variant of Diagonal Distance
        int dx = abs(a.x - b.x);
        int dy = abs(a.y - b.y);
        return max(dx,dy)  + 4*min(dx,dy); //the 4 derives from (14-10) where 14 is the Diagonal Cost and 10 the 'normal cost'
        //return SizeNode*sqrt(pow(a.x* -b.x,2)+pow(a.y-b.y,2)); //euclidean distance
        //return abs(a.x-b.x + a.y-b.y); //manhattan distance
    }

    public void draw(){
        if(start!=null && target!=null && finished){
            drawPath();
        }
    }

    public void drawPath(){
        if(current.predecessor!=null){
            stroke(lineColor.getRGB());
            strokeWeight(sizeNode /2f);
            line(current.x* sizeNode + sizeNode /2f,current.y* sizeNode + sizeNode /2f,current.predecessor.x* sizeNode + sizeNode /2,current.predecessor.y* sizeNode + sizeNode /2);
            current = current.predecessor;
        }
    }
}
