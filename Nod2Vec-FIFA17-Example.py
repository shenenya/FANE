# coding: utf-8
# # Node2Vec showcase
# This notebook is about showcasing the qualities of the node2vec algorithm aswell as my implementation of it which can be found and pip installed through [this link](https://github.com/eliorc/node2vec).
# 
# Check out the related [Medium post](https://medium.com/@eliorcohen/node2vec-embeddings-for-graph-data-32a866340fef).
# 
# Data is taken from https://www.kaggle.com/artimous/complete-fifa-2017-player-dataset-global

# get_ipython().run_line_magic('matplotlib', 'inline')

import warnings
from text_unidecode import unidecode
from collections import deque

warnings.filterwarnings('ignore')

import pandas as pd
from sklearn.manifold import TSNE
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from node2vec import Node2Vec

sns.set_style('whitegrid')

# ### Data loading and pre processing
# Load data
data = pd.read_csv('./FullData.csv', usecols=['Name', 'Club', 'Club_Position', 'Rating'])
# Lowercase columns for convenience
data.columns = list(map(str.lower, data.columns))
# Reformat strings: lowercase, ' ' -> '_' and é, ô etc. -> e, o
reformat_string = lambda x: unidecode(str.lower(x).replace(' ', '_'))
data['name'] = data['name'].apply(reformat_string)
data['club'] = data['club'].apply(reformat_string)
# Lowercase position
data['club_position'] = data['club_position'].str.lower()
# Ignore substitutes and reserves 
data = data[(data['club_position'] != 'sub') & (data['club_position'] != 'res')]
# Fix lcm rcm -> cm cm
fix_positions = {'rcm': 'cm', 'lcm': 'cm', 'rcb': 'cb', 'lcb': 'cb', 'ldm': 'cdm', 'rdm': 'cdm'}
data['club_position'] = data['club_position'].apply(lambda x: fix_positions.get(x, x))
# For example sake we will keep only 7 clubs
clubs = {'real_madrid', 'manchester_utd',
         'manchester_city', 'chelsea', 'juventus',
         'fc_bayern', 'napoli'}
data = data[data['club'].isin(clubs)]
# Verify we have 11 player for each team
assert all(n_players == 11 for n_players in data.groupby('club')['name'].nunique())

# Here comes the ugly part.
# Since we want to put each team of a graph of nodes and edges, I had to hard-code the relationship between the different FIFA 17 formations.
# Also since some formations have the same role (CB for example) in different positions connected to different players,
# I first use a distinct name for each role which after the learning process I will trim so the positions will be the same.
# Finally since position are connected differently in each formation we will add a suffix for the graph presentation and we will trim it also before the Word2vec process
# Example:
# `'cb'` will become `'cb_1_real_madrid'` because it is the first CB, in Real Madrid's formation, and before running the Word2Vec algorithm it will be trimmed to `cb` again
# ### Formations
FORMATIONS = {'4-3-3_4': {'gk': ['cb_1', 'cb_2'],  # Real madrid
                          'lb': ['lw', 'cb_1', 'cm_1'],
                          'cb_1': ['lb', 'cb_2', 'gk'],
                          'cb_2': ['rb', 'cb_1', 'gk'],
                          'rb': ['rw', 'cb_2', 'cm_2'],
                          'cm_1': ['cam', 'lw', 'cb_1', 'lb'],
                          'cm_2': ['cam', 'rw', 'cb_2', 'rb'],
                          'cam': ['cm_1', 'cm_2', 'st'],
                          'lw': ['cm_1', 'lb', 'st'],
                          'rw': ['cm_2', 'rb', 'st'],
                          'st': ['cam', 'lw', 'rw']},
              '5-2-2-1': {'gk': ['cb_1', 'cb_2', 'cb_3'],  # Chelsea
                          'cb_1': ['gk', 'cb_2', 'lwb'],
                          'cb_2': ['gk', 'cb_1', 'cb_3', 'cm_1', 'cb_2'],
                          'cb_3': ['gk', 'cb_2', 'rwb'],
                          'lwb': ['cb_1', 'cm_1', 'lw'],
                          'cm_1': ['lwb', 'cb_2', 'cm_2', 'lw', 'st'],
                          'cm_2': ['rwb', 'cb_2', 'cm_1', 'rw', 'st'],
                          'rwb': ['cb_3', 'cm_2', 'rw'],
                          'lw': ['lwb', 'cm_1', 'st'],
                          'st': ['lw', 'cm_1', 'cm_2', 'rw'],
                          'rw': ['st', 'rwb', 'cm_2']},
              '4-3-3_2': {'gk': ['cb_1', 'cb_2'],  # Man UTD / CITY
                          'lb': ['cb_1', 'cm_1'],
                          'cb_1': ['lb', 'cb_2', 'gk', 'cdm'],
                          'cb_2': ['rb', 'cb_1', 'gk', 'cdm'],
                          'rb': ['cb_2', 'cm_2'],
                          'cm_1': ['cdm', 'lw', 'lb', 'st'],
                          'cm_2': ['cdm', 'rw', 'st', 'rb'],
                          'cdm': ['cm_1', 'cm_2', 'cb_1', 'cb_2'],
                          'lw': ['cm_1', 'st'],
                          'rw': ['cm_2', 'st'],
                          'st': ['cm_1', 'cm_2', 'lw', 'rw']},  # Juventus, Bayern
              '4-2-3-1_2': {'gk': ['cb_1', 'cb_2'],
                            'lb': ['lm', 'cdm_1', 'cb_1'],
                            'cb_1': ['lb', 'cdm_1', 'gk', 'cb_2'],
                            'cb_2': ['rb', 'cdm_2', 'gk', 'cb_1'],
                            'rb': ['cb_2', 'rm', 'cdm_2'],
                            'lm': ['lb', 'cdm_1', 'st', 'cam'],
                            'rm': ['rb', 'cdm_2', 'st', 'cam'],
                            'cdm_1': ['lm', 'cb_1', 'rb', 'cam'],
                            'cdm_2': ['rm', 'cb_2', 'lb', 'cam'],
                            'cam': ['cdm_1', 'cdm_2', 'rm', 'lm', 'st'],
                            'st': ['lm', 'rm', 'cam']},
              '4-3-3': {'gk': ['cb_1', 'cb_2'],  # Napoli
                        'lb': ['cb_1', 'cm_1'],
                        'cb_1': ['lb', 'cb_2', 'gk', 'cm_2'],
                        'cb_2': ['rb', 'cb_1', 'gk', 'cm_2'],
                        'rb': ['cb_2', 'cm_3'],
                        'cm_1': ['cm_2', 'lw', 'lb'],
                        'cm_3': ['cm_2', 'rw', 'rb'],
                        'cm_2': ['cm_1', 'cm_3', 'st', 'cb_1', 'cb_2'],
                        'lw': ['cm_1', 'st'],
                        'rw': ['cm_3', 'st'],
                        'st': ['cm_2', 'lw', 'rw']}}
#Creating the graphs for each team
add_club_suffix = lambda x, c: x + '_{}'.format(c)

graph = nx.Graph()
formatted_positions = set()

def club2graph(club_name, formation, graph):
    club_data = data[data['club'] == club_name]
    club_formation = FORMATIONS[formation]
    club_positions = dict()
    # Assign positions to players
    available_positions = deque(club_formation)
    available_players = set(zip(club_data['name'], club_data['club_position']))
    roster = dict()  # Here we will store the assigned players and positions
    while available_positions:
        position = available_positions.pop()
        name, pos = [(name, position) for name, p in available_players if position.startswith(p)][0]
        roster[name] = pos
        available_players.remove((name, pos.split('_')[0]))
    reverse_roster = {v: k for k, v in roster.items()}
    # Build the graph
    for name, position in roster.items():
        # Connect to team name
        graph.add_edge(name, club_name)
        # Inter team connections
        for teammate_position in club_formation[position]:
            # Connect positions
            graph.add_edge(add_club_suffix(position, club_name),
                           add_club_suffix(teammate_position, club_name))
            # Connect player to teammate positions
            graph.add_edge(name,
                           add_club_suffix(teammate_position, club_name))
            # Connect player to teammates
            graph.add_edge(name, reverse_roster[teammate_position])
            # Save for later trimming
            formatted_positions.add(add_club_suffix(position, club_name))
            formatted_positions.add(add_club_suffix(teammate_position, club_name))
    return graph

teams = [('real_madrid', '4-3-3_4'),
         ('chelsea', '5-2-2-1'),
         ('manchester_utd', '4-3-3_2'),
         ('manchester_city', '4-3-3_2'),
         ('juventus', '4-2-3-1_2'),
         ('fc_bayern', '4-2-3-1_2'),
         ('napoli', '4-3-3')]

graph = club2graph('real_madrid', '4-3-3_4', graph)

for team, formation in teams:
    graph = club2graph(team, formation, graph)

# Node2Vec algorithm
node2vec = Node2Vec(graph, dimensions=20, walk_length=16, num_walks=100, workers=2)
fix_formatted_positions = lambda x: x.split('_')[0] if x in formatted_positions else x
reformatted_walks = [list(map(fix_formatted_positions, walk)) for walk in node2vec.walks]
node2vec.walks = reformatted_walks

model = node2vec.fit(window=10, min_count=1)

# Most similar nodes
# for node, _ in model.most_similar('rw'):
#     # Show only players
#     if len(node) > 3:
#         print(node)
#
# for node, _ in model.most_similar('gk'):
#     # Show only players
#     if len(node) > 3:
#         print(node)
#
# for node, _ in model.most_similar('real_madrid'):
#     print(node)
#
# for node, _ in model.most_similar('paulo_dybala'):
#     print(node)

# Visualization
player_nodes = [x for x in model.vocab if len(x) > 3 and x not in clubs]
embeddings = np.array([model[x] for x in player_nodes])

tsne = TSNE(n_components=2, random_state=7, perplexity=15)
embeddings_2d = tsne.fit_transform(embeddings)

# Assign colors to players
team_colors = {
    'real_madrid': 'lightblue',
    'chelsea': 'b',
    'manchester_utd': 'r',
    'manchester_city': 'teal',
    'juventus': 'gainsboro',
    'napoli': 'deepskyblue',
    'fc_bayern': 'tomato'
}

data['color'] = data['club'].apply(lambda x: team_colors[x])
player_colors = dict(zip(data['name'], data['color']))
colors = [player_colors[x] for x in player_nodes]

figure = plt.figure(figsize=(11, 9))

ax = figure.add_subplot(111)

ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=colors)

# Create team patches for legend
team_patches = [mpatches.Patch(color=color, label=team) for team, color in team_colors.items()]
ax.legend(handles=team_patches)

figure.show()
