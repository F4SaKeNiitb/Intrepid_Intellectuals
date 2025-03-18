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
from src.methods.GARGAML import GARG_AML_node_directed_measures

def process_node(node, G_reduced, G_reduced_und, G_reduced_rev):
    (
        measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
        size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22 
    ) = GARG_AML_node_directed_measures(node, G_reduced, G_reduced_und, G_reduced_rev, include_size=True)
    return (
        node, 
        measure_00, measure_01, measure_02, measure_10, measure_11, measure_12, measure_20, measure_21, measure_22, 
        size_00, size_01, size_02, size_10, size_11, size_12, size_20, size_21, size_22
    )

def score_calculation(dataset):
    directed = True
    path = 'data/edge_data_'+dataset+'.csv'
    G = construct_synthetic_graph(path=path, directed = directed)

    G_reduced = graph_community(G)

    G_reduced_und = G_reduced.to_undirected()
    G_reduced_rev = G_reduced.reverse(copy=True)

    nodes = list(G_reduced.nodes)

    print("Here we go")
    start_time = time.time()

    process_node_partial = partial(process_node, G_reduced=G_reduced, G_reduced_und=G_reduced_und, G_reduced_rev=G_reduced_rev)
    with Pool(cpu_count()) as pool:
        results = list(tqdm(pool.imap(process_node_partial, nodes), total=len(nodes)))

    (
        nodes, 
        measure_00_list, measure_01_list, measure_02_list, measure_10_list, measure_11_list, measure_12_list, measure_20_list, measure_21_list, measure_22_list, 
        size_00_list, size_01_list, size_02_list, size_10_list, size_11_list, size_12_list, size_20_list, size_21_list, size_22_list
    ) = zip(*results)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print(f"Total time taken: {elapsed_time:.2f} seconds")

    data_dict = {
        "node": nodes,
        "measure_00": measure_00_list,
        "measure_01": measure_01_list,
        "measure_02": measure_02_list,
        "measure_10": measure_10_list,
        "measure_11": measure_11_list,
        "measure_12": measure_12_list,
        "measure_20": measure_20_list,
        "measure_21": measure_21_list,
        "measure_22": measure_22_list, 
        "size_00": size_00_list,
        "size_01": size_01_list,
        "size_02": size_02_list,
        "size_10": size_10_list,
        "size_11": size_11_list,
        "size_12": size_12_list,
        "size_20": size_20_list,
        "size_21": size_21_list,
        "size_22": size_22_list
    }

    df = pd.DataFrame(data_dict)
    df.to_csv("results/"+dataset+"_GARGAML_directed.csv", index=False)

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
                            with open('results/time_results_dir.txt', 'a') as f:
                                f.write(string_name + ': ' + str(calc_time) + '\n')
                    if generation_method == 'Erdos-Renyi':
                        m_edges = 0
                        for p_edges in p_edges_list:
                            string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                            start = timeit.default_timer()
                            score_calculation(string_name)
                            end = timeit.default_timer()
                            calc_time = end - start
                            with open('results/time_results_dir.txt', 'a') as f:
                                f.write(string_name + ': ' + str(calc_time) + '\n')

                    if generation_method == 'Watts-Strogatz':
                        for m_edges in m_edges_list:
                            for p_edges in p_edges_list:
                                string_name = 'synthetic_' + generation_method + '_'  + str(n_nodes) + '_' + str(m_edges) + '_' + str(p_edges) + '_' + str(n_patterns)
                                start = timeit.default_timer()
                                score_calculation(string_name)
                                end = timeit.default_timer()
                                calc_time = end - start
                                with open('results/time_results_dir.txt', 'a') as f:
                                    f.write(string_name + ': ' + str(calc_time) + '\n')