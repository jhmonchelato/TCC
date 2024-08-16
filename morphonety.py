from brian2 import *

import neuronDynamics as nd
from scipy.spatial import KDTree
import networkx as nx
import random
import numpy as np

def create_network(tab, n, size=800, axon_len=200, theta=0, seed=45698):
    '''
    
    tab = dataframe do neuronio desejado
    n = numero de neuronios na rede
    size = tamanho da imagem gerada
    axon_len = tamanho do axon artificial que sera gerado para o neurônio
    theta = angulo de rotação do axônio artificial
    standard = utiliza uma seed padrao para que a rede sempre fique igual, 
    apenas mudando o numero de neurônios contidos
    
    retorna o grafo, a imagem da rede gerada e o dicionário com os dados da rede de neuronios
    
    '''
    #random.seed(seed)
    #np.random.seed(seed)

    df = nd.create_tab(tab,axon_len, theta)
   
    img, neurons_dict, img_color, colors = nd.create_net(df, n, size)
    
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=15, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo

    
    network = nx.MultiDiGraph()
    network.add_nodes_from(range(n))
    network.add_edges_from(edges)
    
    return network, img_color, neurons_dict

def create_network_from_swc(neuron_swc, n, size=800, axon_len=200, theta=0, seed=45698):
    '''
    
    neuron_swc = Arquivo swc do neuronio que deseja criar a rede
    n = numero de neuronios na rede
    size = tamanho da imagem gerada
    axon_len = tamanho do axon artificial que sera gerado para o neurônio
    theta = angulo de rotação do axônio artificial
    seed = utiliza uma seed padrao para que a rede sempre fique igual, apenas mudando o numero de neurônios contidos
    
    retorna o grafo, a imagem da rede gerada, dicionário com os dados da rede de neuronios, as arestas do grafo e 
    as cores dos neurônios da imagem
    
    '''
    tab = nd.read_swc(neuron_swc)
    
    df = nd.create_tab(tab,axon_len, theta)
   
    img, neurons_dict, img_color, colors = nd.create_net(df, n, size, seed=seed)
    
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=15, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo

    
    network = nx.MultiDiGraph()
    network.add_nodes_from(range(n))
    network.add_edges_from(edges)
    
    return network, img_color, neurons_dict, edges, colors


def remove_region(region_x, region_y, neurons_dict, n):
    '''
    region_x: tupla com (x1, x2), sendo x1 o inicio da região a ser removida e x2 o final na coordenada x
    region_y: tupla com (y1, y2), sendo y1 o inicio da região a ser removida e y2 o final na coordenada y
    neurons_dict: dicionário com os dados das conectividades da rede
    
    retorna o grafo da rede com os neurônios removidos
    '''
    D = nd.create_D(neurons_dict) #Cria a matriz 2xN com os pixels que possuem conexao 
    tree = KDTree(D) #Criacao da KD tree para obter os pares proximos com query_pairs()
    pairs = tree.query_pairs(r=15, p=2)

    edges = nd.create_edges(D, neurons_dict, pairs) #Cria as arestas do grafo



    edges_raw = sorted(edges)
    
    neurons_to_rm = []
    
    # Encontrando quais neuronios devem ser removidos
    for k,v in neurons_dict.items():
        if k[1] >= region_x[0] and k[1] <= region_x[1] and k[0] >= region_y[0] and k[0] <= region_y[1]:
            for i in v:
                neurons_to_rm.append(i[0])
            
    
    # Remove duplicatas
    neurons_to_rm = set(neurons_to_rm)
    
    # Encontrando o indice da tupla que contém o neuronio a ser removido da lista de arestas
    to_remove = []
    for i in range(len(edges_raw)):
        for indice in edges_raw[i]:
            if indice in neurons_to_rm:
                to_remove.append(edges_raw[i])
               
    # Eliminando duplicatas
    to_remove = list(set(to_remove))


    edges_removed = edges_raw.copy() # Salvando a lista de arestas originais antes de fazer a remocao
    # Removendo os neuronios
    for tupla in to_remove:
        edges_removed.remove(tupla)
    
        
    
    network = nx.MultiDiGraph()
    network.add_nodes_from(range(n))
    network.add_edges_from(edges_removed)

    for node in neurons_to_rm:
        network.remove_node(node)
    
    print(f'Neurônios que foram removidos: {neurons_to_rm}')
    print(f'Conexões mantidas: {edges_removed}')
    
    return network, edges_removed, neurons_to_rm

def symulate_dynamics(network, input_neurons, input_voltage, dyn_params, n):
    '''
    network: Grafo com a rede de neurônios
    input_neurons: Vetor com os neuronios que são carregados durante a simulação
    input_voltage: Voltagem de carregamento dos neurônios selecionados
    dyn_params: Dicionário contendo os dados necessários para a simulação ( {'tau':none, 'threshold':none, 'reset_voltage':none, 'resting_voltage':none} )
    n: Número de neurônios da rede

    retorna o spike monitor (SM) da simulação, número de disparos de cada neurônio (count_each) e número de disparos total da rede (num_spikes)
    '''
    #n = network.number_of_nodes()
    
    start_scope()
    v_rest = dyn_params['resting_voltage']
    thau = dyn_params['tau']
    threshold = dyn_params['threshold']
    reset_voltage = dyn_params['reset_voltage']
    
    eqs = '''
    dv/dt = ((v_rest - v) + V)/tau : volt
    V : volt
    tau : second
    '''
    
    
    G = NeuronGroup(n, eqs, threshold=f'v > {threshold}*volt', reset=f'v = {reset_voltage}*volt', method='exact')

    G.V = ([0]*n)*volt
    for neuron in input_neurons:
        G.V[neuron] = input_voltage*volt
    
    G.v = ([v_rest]*n)
    
    G.tau = ([thau*n])*ms
    for neuron in input_neurons:
        G.tau[neuron] = 10*ms
    
    S = Synapses(G, G, on_pre=f'v_post += {input_voltage/10}*volt')
    
    edges = network.edges
    
    edges_raw = sorted(set(edges))
    edges_raw_form = list(zip(*edges_raw))

    vector_i_raw = edges_raw_form[0]
    vector_j_raw = edges_raw_form[1]
    
    S.connect(i=vector_i_raw, j=vector_j_raw)
    
    M = StateMonitor(G, 'v', record=True)
    PR = PopulationRateMonitor(G)
    SM = SpikeMonitor(G)

    run(25*ms) # Roda a simulacao

    count_each = SM.count # Numero de spikes de cada neuronio
    num_spikes = SM.num_spikes #Numero total de spikes

    #plot de spikes por neuronio

    plt.figure(figsize=(18, 2))

    plt.plot(count_each, '.')
    xticks(np.arange(0, 49, step=1))
    yticks(np.arange(0, 150, step=15))
    plt.title(f"Neuron disp: {input_neurons}")
    plt.xlabel('Neuron Index')
    plt.ylabel('Nro de disparos')
    plt.grid()
    plt.show()

    #Spike Gram

    plt.figure(figsize=(8, 10))

    plt.title('SpikeGram')
    plt.plot(SM.t/ms, SM.i, '.k')
    xticks(np.arange(0, 25, step=1))
    yticks(np.arange(0, 50, step=1))
    plt.title(f"Neuron disp: {input_neurons}")
    plt.xlabel('Time (ms)')
    plt.ylabel('Neuron Index')
    plt.grid()
    plt.show()

    return SM, count_each, num_spikes
    
    
    
    
    
    
    
    
    