import os
import sys
import time
from functools import partial
DIR = "/home/suraj.prasad/idea/smurfing/GARG-AML/"
os.chdir(DIR)
sys.path.append(DIR)


import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
import timeit

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from src.data.graph_construction import construct_IBM_graph, construct_synthetic_graph
from src.utils.graph_processing import graph_community
from src.methods.GARGAML import GARG_AML_node_undirected_measures

# Define a function to process each node
def process_node(node, G_reduced):
    measure_1, measure_2, measure_3, size_1, size_2, size_3 = GARG_AML_node_undirected_measures(node, G_reduced, include_size=True)
    return (node, measure_1, measure_2, measure_3, size_1, size_2, size_3)

def score_calculation(dataset):
    print("====================================")
    print(dataset)
    directed = False
    path = 'data/edge_data_'+dataset+'.csv'
    G = construct_synthetic_graph(path=path, directed = directed)
    
    G_reduced = graph_community(G)
    # Get the list of nodes
    nodes = list(G_reduced.nodes)

    print("Here we go")
    start_time = time.time()
    # Use multiprocessing to process nodes in parallel
    process_node_partial = partial(process_node, G_reduced=G_reduced)
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_node_partial, nodes), total=len(nodes)))

    # Unpack the results
    nodes, measure_1_list, measure_2_list, measure_3_list, size_1_list, size_2_list, size_3_list = zip(*results)

    # End timing
    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total time taken: {elapsed_time:.2f} seconds")

    data_dict = {
        "node": nodes,
        "measure_1": measure_1_list,
        "measure_2": measure_2_list,
        "measure_3": measure_3_list,
        "size_1": size_1_list,
        "size_2": size_2_list,
        "size_3": size_3_list
    }

    df = pd.DataFrame(data_dict)
    df.to_csv("results/"+dataset+"_GARGAML_undirected.csv", index=False)

if __name__ == '__main__':
    n_nodes_list = [100, 10000, 100000] # Number of nodes in the graph
    m_edges_list = [1, 2, 5] # Number of edges to attach from a new node to existing nodes
    p_edges_list = [0.001, 0.01] # Probability of adding an edge between two nodes
    generation_method_list = [
        'Barabasi-Albert', 
        'Erdos-Renyi', 
        'Watts-Strogatz'
        ] # Generation method for the graph
    n_patterns_list = [3, 5] # Number of smurfing patterns to add
    
    for n_nodes in n_nodes_list:
        for n_patterns in n_patterns_list:
            if n_patterns <= 0.06*n_nodes:
                for generation_method in generation_method_list:
                    if generation_method == 'Barabasi-Albert':
                        p_edges = 0
                        for m_edges in m_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            start = timeit.default_timer()
                            score_calculation(string_name)
                            end = timeit.default_timer()
                            calc_time = end - start
                            with open('results/time_results_undir.txt', 'a') as f:
                                f.write(string_name + ': ' + str(calc_time) + '\n')
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            start = timeit.default_timer()
                            score_calculation(string_name)
                            end = timeit.default_timer()
                            calc_time = end - start
                            with open('results/time_results_undir.txt', 'a') as f:
                                f.write(string_name + ': ' + str(calc_time) + '\n')

                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                start = timeit.default_timer()
                                score_calculation(string_name)
                                end = timeit.default_timer()
                                calc_time = end - start
                                with open('results/time_results_undir.txt', 'a') as f:
                                    f.write(string_name + ': ' + str(calc_time) + '\n')