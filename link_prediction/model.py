import random
import numpy as np
from sklearn.metrics import roc_auc_score
import pandas
from tqdm import tqdm
import node2vec
import networkx as nx
from gensim.models import Word2Vec
from sklearn.model_selection import ParameterGrid
import time

total_unknown = 0

def get_neighbourhood_score(local_model, node1, node2, is_test=False):
    # Provide the plausibility score for a pair of nodes based on your own model.
    global total_unknown
    try:
        vector1 = local_model.wv.vectors[local_model.wv.index2word.index(node1)]
        vector2 = local_model.wv.vectors[local_model.wv.index2word.index(node2)]
        return np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
    except:
        if is_test:
            total_unknown += 1
        return 0.5 


def get_G_from_edges(edges):
    edge_dict = dict()
    # calculate the count for all the edges
    for edge in edges:
        edge_key = str(edge[0]) + '_' + str(edge[1])
        if edge_key not in edge_dict:
            edge_dict[edge_key] = 1
        else:
            edge_dict[edge_key] += 1
    tmp_G = nx.DiGraph()
    for edge_key in edge_dict:
        weight = edge_dict[edge_key]
        # add edges to the graph
        tmp_G.add_edge(edge_key.split('_')[0], edge_key.split('_')[1])
        # add weights for all the edges
        tmp_G[edge_key.split('_')[0]][edge_key.split('_')[1]]['weight'] = weight
    return tmp_G


def get_AUC(model, true_edges, false_edges):
    true_list = list()
    prediction_list = list()
    for edge in true_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(1)
        prediction_list.append(tmp_score)

    for edge in false_edges:
        tmp_score = get_neighbourhood_score(model, str(edge[0]), str(edge[1]))
        true_list.append(0)
        prediction_list.append(tmp_score)
    y_true = np.array(true_list)
    y_scores = np.array(prediction_list)
    return roc_auc_score(y_true, y_scores)




# Start to load the train data

train_edges = list()
raw_train_data = pandas.read_csv('train.csv')
for i, record in raw_train_data.iterrows():
    train_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the train data.')

# Start to load the valid data

valid_positive_edges = list()
valid_negative_edges = list()
raw_valid_data = pandas.read_csv('valid.csv')
for i, record in raw_valid_data.iterrows():
    if record['label']:
        valid_positive_edges.append((str(record['head']), str(record['tail'])))
    else:
        valid_negative_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the valid data.')

# Start to load the test data

test_edges = list()
raw_test_data = pandas.read_csv('test.csv')
raw_test_data['label'] = False
raw_test_data['score'] = .0
raw_test_data['head'] = raw_test_data['head'].apply(np.int64)
raw_test_data['tail'] = raw_test_data['tail'].apply(np.int64)
for i, record in raw_test_data.iterrows():
    test_edges.append((str(record['head']), str(record['tail'])))

print('finish loading the test data.')
directed = True
history = []
param_grid = {
    'dimension': [300],
    'iterations': [30],
    'num_walks': [45],
    'num_workers': [2000],
    'p': [50],
    'q': [1],
    'walk_length': [200],
    'window_size': [200],
}
grid = ParameterGrid(param_grid)
max_of = 1
best_model = None
best_auc = 0
best_params = None

total_trials = 0
for _ in grid:
    total_trials += 1

for idx, params in enumerate(grid):
    st = time.time()
    for _ in range(max_of):
        # write code to train the model here
        # Create a node2vec object with training edges
        G = node2vec.Graph(get_G_from_edges(train_edges),
                           directed, params['p'], params['q'])
        # Calculate the probability for the random walk process
        G.preprocess_transition_probs()
        # Conduct the random walk process
        walks = G.simulate_walks(params['num_walks'], params['walk_length'])
        # Train the node embeddings with gensim word2vec package
        model = Word2Vec(walks, size=params['dimension'],
                         window=params['window_size'], min_count=0, sg=1,
                         workers=params['num_workers'],
                         iter=params['iterations'])

        # evaluate
        tmp_AUC_score = get_AUC(model, valid_positive_edges,
                                valid_negative_edges)
        history.append(params)
        history[-1]['auc_score'] = tmp_AUC_score


        if tmp_AUC_score > best_auc:
            best_model = model
            best_auc = tmp_AUC_score
            best_params = params

    print("Progress: %d/%d" % (idx + 1, total_trials))

    print("Params:")
    for k in params:
        if k == 'auc_score':
            print("\t%s = %f" % (k, params[k]))
        else:
            print("\t%s = %d" % (k, params[k]))
    print("Duration:", time.time() - st, "seconds")
    print()


print("Best params (max of 5 models):")
for k in best_params:
    if k == 'auc_score':
        print("\t%s = %f" % (k, best_params[k]))
    else:
        print("\t%s = %d" % (k, best_params[k]))

print('\n')
print('Best accuracy:', best_auc)
print('end')


for i, edge in enumerate(test_edges):
    raw_test_data.at[i, 'score'] = get_neighbourhood_score(best_model, str(edge[0]), str(edge[1]), True)
    if raw_test_data.loc[i]['score'] >= 0.5:
        raw_test_data.at[i, 'label'] = True

print(total_unknown, "/", len(test_edges), "=", total_unknown/len(test_edges))
raw_test_data.to_csv("test_pred.csv", index=False, columns=['head', 'label', 'score', 'tail'])

# df = pandas.DataFrame(history)
# df.to_csv("grid_search_last2.csv", index=False, quoting=csv.QUOTE_ALL)
