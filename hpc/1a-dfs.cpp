#include <iostream>
#include <vector>
#include <unordered_map>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    unordered_map<int, vector<int>> adj;
    unordered_map<int, bool> visited;

public:
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void parallelDfs(int node) {
        #pragma omp critical
        {
            if (visited[node]) {
                return;
            }
            visited[node] = true;
            cout << node << " ";
        }
        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            if (!visited[neighbor]) {
                parallelDfs(neighbor);
            }
        }
    }
};

int main() {
    int V, E;
    cout << "Enter Number of vertices :- ";
    cin >> V;
    Graph G;
    cout << "Enter Number of Edges: -";
    cin >> E;
    cout << "Enter Edges (u-v)" << endl;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        G.addEdge(u, v);
    }
    int startNode;
    cout << "Enter Starting Node ";
    cin >> startNode;
    G.parallelDfs(startNode);
    return 0;
}
