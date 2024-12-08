import streamlit as st
import networkx as nx
import numpy as np
import itertools
import pulp as pl

import matplotlib.pyplot as plt

st.title("Graph Coloring")
num_nodes = st.number_input("Enter the number of nodes", min_value=5, step=1)
adj_matrix_input = st.text_area("Enter the adjacency matrix (comma-separated rows)")

if st.button("Generate Graph"):
    adj_matrix = []
    try:
        rows = adj_matrix_input.split('\n')
        for row in rows:
            adj_matrix.append(list(map(int, row.split(','))))
        
        G = nx.Graph()
        G.add_nodes_from(range(1, num_nodes + 1))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i + 1, j + 1)
        st.session_state['G'] = G
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray',edgecolors='black')
        st.pyplot(plt.gcf())
    except Exception as e:
        st.write("Invalid input.")
        st.write("Generated random Graph.")
        def generate_random_adj_matrix(num_nodes):
            matrix = np.random.randint(0, 2, size=(num_nodes, num_nodes))
            np.fill_diagonal(matrix, 0)
            return matrix

        adj_matrix = generate_random_adj_matrix(num_nodes)
        G = nx.Graph()
        G.add_nodes_from(range(1, num_nodes + 1))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                if adj_matrix[i][j] == 1:
                    G.add_edge(i + 1, j + 1)

        st.session_state['G'] = G
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray', edgecolors='black')
        st.pyplot(plt.gcf())


def solve1(G):
    def dsatur(graph):
        color_map = {}
        saturation_degree = {node: 0 for node in graph.nodes()}
        degree = dict(graph.degree())
        
        def max_saturation_node():
            max_sat_deg = -1
            max_deg = -1
            max_node = None
            for node in graph.nodes():
                if node not in color_map:
                    current_saturation = saturation_degree[node] if saturation_degree[node] > 0 else degree[node]
                    if (current_saturation > max_sat_deg) or \
                    (current_saturation == max_sat_deg and degree[node] > max_deg):
                        max_sat_deg = current_saturation
                        max_deg = degree[node]
                        max_node = node
            return max_node

        while len(color_map) < len(graph.nodes()):
            node = max_saturation_node()
            neighbor_colors = set(color_map[neighbor] for neighbor in graph.neighbors(node) if neighbor in color_map)
            for clr in range(1, len(graph.nodes()) + 1):
                if clr not in neighbor_colors:
                    color_map[node] = clr
                    break
            for neighbor in graph.neighbors(node):
                if neighbor not in color_map:
                    saturation_degree[neighbor] += 1

        cliques = list(nx.find_cliques(graph))
        max_clique_size = max(len(clique) for clique in cliques)
        print("Max clique size:", max_clique_size)
        
        return color_map
    color_map = dsatur(G)
    colors = [color_map[node] for node in G.nodes()]
    st.write("number of colors used:", len(set(colors)))
    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, edge_color='gray', cmap=plt.cm.Pastel1,edgecolors='black')
    st.pyplot(plt.gcf())
    
def solve(G):
    print("Solving...")
    P4s = []
    for u, v, w, x in itertools.combinations(G.nodes(), 4):
        # Vrifier si les sommets forment un chemin de 4 sommets
        if G.has_edge(u, v) and G.has_edge(v, w) and G.has_edge(w, x):
            P4s.append((u, v, w, x)) # Ajouter le P4 la liste
    st.write(f"Number of P4s: {len(P4s)}")
    def perfectly_order_graph(G, P4s):
        position = pl.LpVariable.dicts("position", G.nodes(), 1, len(G.nodes()) , cat='Integer')
        a=pl.LpVariable.dicts("a",[(i,j) for i in range(1,len(G.nodes())) for j in range(i+1, len(G.nodes())+1)],0,1,cat='Binary')

        y = pl.LpVariable.dicts("y", [(u,v) for u,v in G.edges()], 0, 1, cat='Binary')


        problem = pl.LpProblem("ordering", pl.LpMaximize)
        epsilon = 0.00001
        M=10000
        problem += 0
        problem += position[4] == 1

        for i in range(1,len(G.nodes())):
            for j in range(i+1, len(G.nodes())+1):
                problem += position[i] - position[j] +M*a[(i,j)] >= 1
                problem += position[i] - position[j] +M*a[(i,j)] <=M-1
                # problem += position[i] - position[j] +M*a[(i,j)] <= -epsilon+M*a[(i,j)]
                # problem += position[i] - position[j] +M*a[(i,j)] >= epsilon -M*(1-a[(i,j)])
                # problem += position[i] != position[j]

        # for each P4, add a constraint that the nodes are ordered correctly
        for P4 in P4s:
            u, v, w, x = P4
            problem += position[u]-position[v] >= 1-(1-y[(u,v)])*M
            problem += position[x]-position[w] >= 1-(1-y[(w,x)])*M
            problem += y[(u,v)]+y[(w,x)] >= 1

        problem.solve()

        if pl.LpStatus[problem.status] == "Optimal":
            for node in G.nodes():
                print(f"Node {node} is in position {pl.value(position[node])}")
            perfectly_ordered = True
            order=[pl.value(position[node]) for node in G.nodes()]
        else:
            perfectly_ordered = False
            order=[]
        return perfectly_ordered, order
    def greedy_coloring(G, order):
        color_map = {}
        for node in sorted(G.nodes(), key=lambda x: order[x-1]):
            available_colors = set(range(len(G.nodes())))
            for neighbor in G.neighbors(node):
                if neighbor in color_map:
                    available_colors.discard(color_map[neighbor])
            color_map[node] = min(available_colors)
        return color_map


    print('finding perfect ordering')
    is_perfectly_ordered, order = perfectly_order_graph(G, P4s)
    if is_perfectly_ordered:
        st.write('Graph is perfectly orderable')
        st.write('Order found with linear programming:', order)
        color_map = greedy_coloring(G, order)
        colors = [color_map[node] for node in G.nodes()]

        st.write("number of colors used:", len(set(colors)))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color=colors, node_size=500, edge_color='gray', cmap=plt.cm.Pastel1,edgecolors='black')
        st.pyplot(plt.gcf())
    else:
        st.write('Graph is not perfectly ordered')
        st.write('try DSatur algorithm :')

if st.button("Load Example Graph"):
    G = nx.Graph()
    G.add_nodes_from([1, 2, 3, 4, 5, 6, 7, 8, 9])
    G.add_edges_from([(1, 3), (1, 4), (1, 7), (1, 9), (2, 4), (1, 5), (2, 8), (2, 9),
                      (3, 5), (3, 9), (3, 8), (4, 6), (4, 7), (5, 6), (5, 7), (6, 8), 
                      (6, 9), (7, 8), (7, 9)])
    st.session_state['G'] = G

    pos = nx.spring_layout(G)
    nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=500, edge_color='gray')
    st.pyplot(plt.gcf())




if st.button("Coloring for perfectly ordered graph"):
    G = st.session_state['G']
    solve(G)
if st.button("Coloring using DSatur"):
    G = st.session_state['G']
    solve1(G)

