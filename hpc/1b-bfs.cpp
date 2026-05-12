#include <vector>
#include <unordered_map>
#include <iostream>
#include <queue>
#include <omp.h>
using namespace std;

class Graph {
    int V;
    unordered_map<int, vector<int>> adj;

public:
    void addEdge(int u, int v) {
        adj[u].push_back(v);
        adj[v].push_back(u);
    }

    void ParallelBfs(int start) {
        unordered_map<int, bool> visited;
        queue<int> q;
        visited[start] = true;
        q.push(start);

        while (!q.empty()) {
            int size = q.size();
            #pragma omp parallel for
            for (int i = 0; i < size; i++) {
                int node;
                #pragma omp critical
                {
                    node = q.front();
                    q.pop();
                    cout << " " << node;
                }
                for (int neighbor : adj[node]) {
                    #pragma omp critical
                    {
                        if (!visited[neighbor]) {
                            visited[neighbor] = true;
                            q.push(neighbor);
                        }
                    }
                }
            }
        }
        cout << endl;
    }
};

int main() {
    int V, E;
    cout << "Enter Number of Vertices:- ";
    cin >> V;
    Graph g;
    cout << "Enter Number of Edges : - ";
    cin >> E;
    cout << "Enter Edges (u v )" << endl;
    for (int i = 0; i < E; i++) {
        int u, v;
        cin >> u >> v;
        g.addEdge(u, v);
    }
    int startNode;
    cout << "Enter Start Node: - ";
    cin >> startNode;
    g.ParallelBfs(startNode);
    return 0;
}