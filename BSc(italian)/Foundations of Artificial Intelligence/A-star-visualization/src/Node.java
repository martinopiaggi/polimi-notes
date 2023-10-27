public class Node implements Comparable<Node>{
    final public int x;
    final public int y;
    public boolean free;
    public int cost;
    public boolean visited;
    public Node predecessor;

    public Node(int x,int y,boolean free, int cost) {
        this.x = x;
        this.y = y;
        this.free = free;
        this.cost = cost;
        this.visited = false;
        predecessor = null;
    }

    public boolean equals(Node o){
        if(o.x == this.x && o.y == this.y)return true;
        else return false;
    }

    @Override
    public int compareTo(Node a){
        if(this.cost>a.cost)return 1;
        else if(this.cost<a.cost)return 0;
        else return -1;
    }

}
