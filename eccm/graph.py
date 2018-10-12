from graphviz import Digraph
import numpy as np

class Node():
    def __init__(self, name):
        '''
        Next is a dictionary with {next_node: dist} as its k-v.
        '''
        self.name = name
        self.next = {}
        
    def __getitem__(self, child):
        return self.next[child]
        
    def __setitem__(self, child, dist):
        '''
        Add a new child and its corresponding dist to self.next.
        '''
        assert dist >= 0
        self.next[child] = dist
        
    def __delitem__(self, child):
        del self.next[child]
        
    def __str__(self):
        return f'Node {self.name}'
    
    def __repr__(self):
        return f'Node {self.name}'
            
    def children(self):
        '''
        Return a list of child nodes.
        '''
        return list(self.next.keys())
            
class Graph():
    def __init__(self, nodes, edges, dists):
        '''
        Inputs:
            edges: List of edges, with each edge being 'X → Y'.
            dists: Corresponding dist value for edge 'X → Y'.
        '''
        # Create a dictionary of nodes
        self.nodes = {node_id: Node(node_id) for node_id in nodes}
        
        # Populate graph
        for edge, dist in zip(edges, dists):
            # Get source and destination nodes
            src, dst = edge.split(' → ')
                
            # Add child node to `src`
            self.nodes[src][self.nodes[dst]] = dist
            
    def __repr__(self):
        g = Digraph()
        
        for node_id, node in self.nodes.items():
            g.node(node_id)
            for child in node.children():
                g.edge(node.name, child.name, label=str(node[child]))
                
        g.engine = 'circo'
                
        display(g)
        return ''
    
    def __getitem__(self, node_name):
        return self.nodes[node_name]
    
    def adj_mat(self):
        '''
        Return an adjacency matrix equivalent of the graph 
        and the list of nodes.
        '''
        # Get number of nodes in graph
        N = len(self.nodes)
        
        # Initialize a square array of zeros
        hm = np.zeros((N, N))
        
        # Get sorted list of nodes
        node_list = sorted(self.nodes.keys())
        
        for i, node_id in enumerate(node_list):
            for child in self[node_id].children():
                hm[node_list.index(child.name), i] = 1
                
        return hm, node_list
            
    
    def prune(self, verbose=False):
        '''
        Check and prune indirect causalities in graph.
        '''
        for src in self.nodes:
            for dst in self.nodes:
                if src == dst:
                    continue
                    
                # Find shortest indirect distances from `src` to `dst`
                shortest_dists = self.shortest_indirect_dist(src, dst)

                # Prune direct connection if shortest indirect distance
                # is equal to or less than direct distance
                if self[dst] in self[src].children() and self[src][self[dst]] >= shortest_dists[dst]:
                    del self[src][self[dst]]
                    
                    if verbose:
                        print(f'{src} → {dst} pruned')
        
    def shortest_indirect_dist(self, src, dst):
        '''
        Perform Dijkstra's algorithm.
        Algorithm is modified to exclude direct connectino from `src` to `dst`.
        
        Return a dictionary of {node_name: dst} relative to `src`.
        '''
        def extract_min(dists, in_region):
            '''
            Return the node_id of a node in the unknown region
            with the minimum distance.
            '''
            # Find node with minimum distance
            min_dist = _INF + 1
            min_node_id = None
            
            for node_id, dist in dists.items():
                if dist < min_dist and not in_region[node_id]:
                    min_node_id = node_id
                    min_dist = dist
            
            return self[min_node_id]
        
        _INF = 1e8 # Proxy distance for infinity
        
        # If direct edge exists from `src` to `dst`, set value to infinity
        if self[dst] in self[src].children():
            ori_direct_dist = self[src][self[dst]]
            self[src][self[dst]] = _INF
            
        # Create lists to track distances and 
        # whether a node is in the known region
        dists = {node: _INF for node in self.nodes.keys()}
        in_region = {node: False for node in self.nodes.keys()}
        
        # Set `src` node distance
        dists[src] = 0
        
        # Iterate through nodes until all nodes are 
        # in the known region
        while not all([val for _, val in in_region.items()]):
            u = extract_min(dists, in_region)
            
            # Set u to be part of the known region
            in_region[u.name] = True
            
            # Iterate through all child nodes, v
            for v in u.children():
                if dists[v.name] > dists[u.name] + u[v]:
                    dists[v.name] = dists[u.name] + u[v]
            
        # Undo changes to direct edge dist
        if 'ori_direct_dist' in locals():
            self[src][self[dst]] = ori_direct_dist
            
        return dists

