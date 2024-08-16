import pandas as pd
import numpy as np
import random
import time
from skimage.draw import line
from skimage.color import hsv2rgb
from collections import defaultdict
from scipy import ndimage as ndi
import networkx as nx

def read_swc(arquivo):
    '''
    read_swc()

    Funcao para ler o arquivo SWC (dado um arquivo de um neuronio em formato SWC, um dataframe é retornado com os dados formatados)

    arquivo = nome do arquivo SWC
    '''
    df = pd.read_csv(arquivo, header=None)
    colunas = ['knot','tipo','x', 'y','z', 'raio','pai']
    remove = []
    for i, row in df.iterrows():
        linha = row[0].split()
        if linha[0] == '#' or linha[0] == '#The':
            remove.append(i)
        if linha[0] == 1:
            break
            
    df = pd.read_csv(arquivo, skiprows=remove, header=None, sep=' ')
    df.drop([0],axis=1, inplace=True)
    df.columns = colunas
    df.loc[0, 'pai'] = 1
    
    return df

'''
def muda_res(df, k):
    
    muda_res()

    Alteracao da resolucao da imagem do neurônio

    df = dataframe com os dados dos segmentos do neuronio
    k = constante de aumento/diminuicao da imagem
    
    
    df['x'] = df['x']/k
    df['y'] = df['y']/k
    df = df.astype({"x": int, "y": int})
    
    return df
'''

def draw_axon(df, L, theta):
    '''
    draw_axon()

    Remove o axonio original e adiciona as coordenadas de um axonio artificial no daframe

    df=dataframe, 
    L=comprimento do axonio, 
    theta=angulo de direcao do axonio a partir do soma
    '''

    #Salvando os indices originais para coloca-los apos a remocao
    df['index'] = df.index
   
    #removendo o axonio para posteriormente desenhar um artificial
    axons_index = df[df["tipo"]==2].index
    df = df.drop(axons_index)
    
    df.set_index('index')
        
    #graus para radianos
    theta = np.radians(theta)
    
    #Obtencao das posicoes da imagem novamente apos o deslocamento
    xc, yc = df.loc[0, ['x', 'y']].values # centro
    x = int(L*np.cos(theta) + xc) # x final
    y = int(L*np.sin(theta) + yc) # y final
    
    #Obtem as coordenadas referentes ao axonio artificial
    rr, cc = line(yc, xc, y, x)
    
    #Adiciona o axonio artifical ao final do dataframe
    flag = True
    for row, col in zip(rr, cc):
        if flag:
            df.loc[df.index[-1] + 1] = [2, col, row, 0, df.index[-1] + 1]
            flag = False
        else:    
            df.loc[df.index[-1] + 1] = [2, col, row, df.index[-1], df.index[-1] + 1]
        
    return df


def create_tab(df, L=50, theta=0): 
    '''
    create_tab()

    Criacao do dataFrame formatado para posteriormente criar a imagem

    df= dataframe que foi criado ao ler o arquivo SWC
    L = Comprimento do axonio artificial,
    theta = angulacao do axonio artificial
    '''
    
    df = df[['tipo', 'x', 'y', 'pai']]
    df = df.astype({"tipo": np.uint8, "x": int, "y": int, "pai": int})
    df['pai'] = df['pai'] - 1
    
    #Desenho do axonio artificial
    df = draw_axon(df, L, theta)

    #shift da imagem (menor valor ser o '0')
    shift_x = min(df['x'])
    shift_y = min(df['y'])
    df['x'] = df['x'] - shift_x
    df['y'] = df['y'] - shift_y

    
    #Rotacao de 90 graus
    df['y'] = max(df['y']) - df['y']
    
    #Definicao dos tons de cinza de cada pixel de acordo com seu tipo (soma, axon, dendrite)
    
    df.loc[df.tipo == 1,'tipo']= 1
    df.loc[df.tipo == 2,'tipo']= 2
    df.loc[df.tipo == 3,'tipo']= 1
    df.loc[df.tipo == 4,'tipo']= 1
    df.loc[df.tipo == 5,'tipo']= 1
    df.loc[df.tipo == 6,'tipo']= 1
    df.loc[df.tipo == 7,'tipo']= 1

    
    return df


def create_img(df, k=1):
    '''
    create_img()

    Criacao da imagem recebendo o dataFrame e a resolucao escolhida

    df = dataframe
    k = constante de aumento/diminuicao da imagem

    '''
    #definicao da resolucao (k==1 default)
    if (k != 1):
        df['x'] = df['x']/k
        df['y'] = df['y']/k
        df = df.astype({"x": int, "y": int})
    
    #criacao da matriz de zeros
    imagem = np.zeros([max(df['y']) + 1,max(df['x']) + 1], dtype=np.uint8)
    
    #imagem[0:max(df['y']) + 1, 0:max(df['x']) + 1] = 255 #Necessario se quiser pintar neuronio unico
    
    #indexacao dos pixels
    for index, row in df.iterrows():
        imagem[row['y'], row['x']] = row['tipo']
        
    # Preenche os buracos deixados na imagem devido a normalizacao
    imagem = draw_lines(df, imagem)
    
    
    #imagem = imagem - 1
    #imagem = ndi.binary_dilation(imagem, iterations=1).astype(np.uint8) #Dilatacao da imagem
    
    
    return imagem

def draw_lines(df, img):
    '''
    draw_lines()

    Completa os 'buracos' que ficam na imagem devido a normalizacao dos valores

    df = dataframe
    img = imagem
    '''
    #color_dend = [0, 0, 0]
    #color_axon = [255, 0, 0]
    
    # Itera pelo dataframe e liga o 'filho' com o 'pai' usando a funcao line do pacote skimage.draw 
    for index, row in df.iterrows():
        x1, y1 = row['x'], row['y']
        
        pai = row['pai']
        linha_pai = df.loc[pai]
        x2, y2 = int(linha_pai['x']), int(linha_pai['y'])
        
        rr, cc = line(y2, x2, y1, x1)
        
        img[rr, cc] = 1
        #img[rr, cc] = color_dend #Pintar Dendrito
        
        
    # Tratamento para pixel de axonio ter valor 2
    df_aux = df.loc[df.tipo == 2] 
    cc, rr = df_aux['x'], df_aux['y']
    rr = list(rr)
    cc = list(cc)
    
    img[rr, cc] = 2
    #img[rr, cc] = color_axon #Pintar Axonio
    
    return img


def rot_df(df):
    '''
    rot_df()

    Faz a rotacao do neuronio em um angulo aleatorio

    df = dataframe
    '''
    #Obtem um valor aleatorio para o angulo de rotacao
    theta = random.uniform(0, 2*np.pi) #angulo aleatorio de rotacao
    
    #matriz de rotacao
    M = np.array([[np.cos(theta), np.sin(theta)],[-(np.sin(theta)), np.cos(theta)]]) 
    
    df_coord = np.array(df[['x', 'y']])
    
    rot_r = M.dot(df_coord.transpose()).transpose().astype('int32')
    
    rot_r -= np.min(rot_r, axis=0)
    
    df['x'] = rot_r[:, 0]
    df['y'] = rot_r[:, 1]
        
    return df


def create_net(df, n, tam_img, seed=45698):
    '''
    create_net()

    Cria uma rede com neuronios iguais, retornando uma imagem da rede e um dict com as coordenadas que possuem conexao

    df = dataframe
    n = numero de neuronios na rede
    tam_img = tamanho da imagem da rede
    '''
    random.seed(seed)
    np.random.seed(seed)
    
    neurons_dict = defaultdict(list)
    net_img = np.zeros([tam_img,tam_img], dtype=np.uint8)
    
    net_img_color = np.zeros([tam_img,tam_img, 3], dtype=np.uint8)
    #coords_neurons = []
    colors = []
    
    for neuronio in range(n):
        color_hue = np.random.randint(0, 256)
        color_sat = np.random.randint(150, 256)
        color_value = np.random.randint(200, 256)
        
        color = [color_hue, color_sat, color_value]
        colors.append(color)
        df = rot_df(df)
        img = create_img(df)
        
        #img = ndi.binary_dilation(img, iterations=1).astype(np.uint8) # Dilatacao da imagem
        
        #Delocamento em x
        rand_x = random.randint(0, tam_img-len(img[0])) 
        #Delocamento em y
        rand_y = random.randint(0,tam_img-len(img))
        
        
        
        coords = np.nonzero(img)
        coords_x = coords[1] + rand_x
        coords_y = coords[0] + rand_y
        net_img[coords_y, coords_x] = net_img[coords_y, coords_x] + img[coords[0], coords[1]]
        
        net_img_color[coords_y, coords_x] = color
        
        #coords_neurons.append([coords_y, coords_x])
    
        for coord_x, coord_y in zip(coords_x, coords_y):
            coord_x_real = coord_x - rand_x #pega a coordenada x original do neuronio, antes do deslocamento
            coord_y_real = coord_y - rand_y #pega a coordenada y original do neuronio, antes do deslocamento
      
            
            indice = img[coord_y_real, coord_x_real]
            
            neurons_dict[(coord_y, coord_x)].append((neuronio, indice))
        
               
            
    net_img_color = hsv2rgb(net_img_color)
    
    return net_img, neurons_dict, net_img_color, colors


def create_D(neurons_dict):
    '''
    Cria a matriz 2xN, onde representa os pixels com overlap. 
    exemplo: dict = {(12, 25): [(2,1), (5,2)],
                     (12, 26): [(2,1), (5,2)],
                     (13, 25): [(2,1), (5,2)],
                     (23, 45): [(4,2), (8,1)]}
                     
    D = [[12,25],
         [12,26],
         [13,25],
         [23,45]]
    '''
    D = []
    for k,v in neurons_dict.items():
        if(len(v)>1):
            for i in range(0,len(v)-1):
                for j in range(i+1, len(v)):
                    if(v[i][1] != v[j][1]):
                        D.append(k)
    
    return D


def create_edges(D, neurons_dict, pairs):
    '''
    create_edges()
    
    Cria o conjunto de aresta utilizado para criar o grafo final
    
    D = Matriz Nx2 com os pixels que possuem sobreposicao
    neurons_dict = Dicionario com os neuronios do espaco amostral
    pairs = Neuronios agrupados que estao a uma distancia r um do outro (tree.query_pairs(r))
    '''
    
    #Criacao de grafo auxiliar para usar os componentes conexos e outras variaveis necessarias para o funcionamento da funcao
    graph1 = nx.Graph()
    graph1.add_nodes_from(range(len(D)))
    graph1.add_edges_from(pairs)
    comps = nx.connected_components(graph1)
    edges = []
    
    # Atraves dos componentes conexos é criada uma lista com os neuronios que possuem conexao
    for comp in comps:
        lista = []
        #lista_pixels = []
        for ind_pixel in comp:
            pixel = D[ind_pixel]
            lista.extend(neurons_dict[pixel])
            #lista_pixels.append(pixel)

        lista = list(set(lista))
        #print(lista_pixels, lista)
        
        #A partir da lista é criado o conjunto de aresta que será usado para criar o grafo final
        for ind1 in range(0, len(lista)-1):
            for ind2 in range(ind1+1, len(lista)):
                node1 = lista[ind1]
                node2 = lista[ind2]
                if(node1[1] != node2[1] and node1[0] != node2[0]):
                    edges.append((node1[0], node2[0]))
        
    return edges