from dash import Dash, Input, Output, State, callback, no_update, ctx, jupyter_dash, html, dcc
import dash_cytoscape as cyto
import dash_bootstrap_components as dbc
from sklearn.model_selection import KFold
from flask import session
from flask_session import Session
import networkx as nx
import pandas as pd
import numpy as np
# In[21]:
import os
os.environ["OMP_NUM_THREADS"] = '4'

import pybnesian as pbn
from uuid import uuid4
import sys
import math
import itertools
import plotly
import base64
import io
import re


jupyter_dash.default_mode = "external"
cyto.load_extra_layouts()
app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    requests_pathname_prefix='/Model/LearningFromData/ISLDash/',
    suppress_callback_exceptions=True
)

server = app.server

# Configure server-side session for Flask
server.config["SESSION_TYPE"] = "filesystem"  # Use file-based sessions
server.config["CACHE_TYPE"] = "FileSystemCache"
server.config["CACHE_DIR"] = "./test"

Session(server)

default_stylesheet = [
    # Define node styles
    {
        'selector': 'node',
        'style': {
            # Scale size between 10 and 20
            'width': 'mapData(size, 1, 100, 10, 20)',
            'height': 'mapData(size, 1, 100, 10, 20)',
            'background-color': '#666',
            'label': 'data(id)',
            'font-size': 'mapData(label_size, 0, 100, 10, 20)',

        }
    },
    # Define edge styles
    {
        'selector': 'edge[directed = "True"]',
        'style': {
            'width': 2,
            'line-color': '#ccc',
            'curve-style': 'bezier',
            'target-arrow-shape': 'triangle',
            'target-arrow-color': '#9c9c9c',  # Arrowhead color
            'source-arrow-color': '#9c9c9c',
        }
    },
    {
        'selector': 'edge[directed = "False"]',
        'style': {
            'width': 2,
            'line-color': '#ccc',
            'curve-style': 'bezier',
            'target-arrow-shape': 'none',  # Edges have no arrowhead
        }
    },
    {
        'selector': 'node[factor_type = "DiscreteFactor"]',
        'style': {
            'background-color': '#bff5ff',  # color for discrete node
            'width': 'mapData(size, 0, 100, 10, 20)',
            'height': 'mapData(size, 0, 100, 10, 20)',
        }
    },
    {
        'selector': 'node[factor_type = "LinearGaussianFactor"]',
        'style': {
            'background-color': '#f8ea8c',  # color for gaussian node
            'width': 'mapData(size, 0, 100, 10, 20)',
            'height': 'mapData(size, 0, 100, 10, 20)',
        }
    },
    {
        'selector': 'node[factor_type = "CKDEFactor"]',
        'style': {
            'background-color': '#E98980',  # color for kde node
            'width': 'mapData(size, 0, 100, 10, 20)',
            'height': 'mapData(size, 0, 100, 10, 20)',
        }
    }

]

operators = {
    "greaterThanOrEqual": "ge",
    "lessThanOrEqual": "le",
    "lessThan": "lt",
    "greaterThan": "gt",
    "notEqual": "ne",
    "equals": "eq",
}

layout_dict = {'Circular': nx.drawing.circular_layout,
               'Fruchterman-Reingold': nx.drawing.spring_layout,
               'Graphviz neato':None,
               'Kamada-Kawai': nx.drawing.kamada_kawai_layout,
               'Shell': nx.drawing.shell_layout,
               'Spectral': nx.drawing.spectral_layout,
               'Spiral': nx.drawing.spiral_layout}

column_definitions_dict = {
    'score_and_search' : [
                    {"field": "Operation"},
                    {"field": "Source"},
                    {"field": "Target"},
                    {"field": "Motivation", "headerName": "Motivation", "filter": "agNumberColumnFilter",
                        'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
                ],
    'node_type_selection' : [
                    {"field": "Operation"},
                    {"field": "Node"},
                    {"field": "Type"},
                    {"field": "Motivation", "headerName": "Motivation", "filter": "agNumberColumnFilter",
                        'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
                ],
    'adjacency_search' : [
                    {"field": "Operation"},
                    {"field": "Node1"},
                    {"field": "Node2"},
                    {"field": "Separating set"},
                    {"field": "p-value", "headerName": "p-value", "filter": "agNumberColumnFilter",
                     'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
                ],
    'v_structure_determination' : [
                    {"field": "Operation"},
                    {"field": "Parents"},
                    {"field": "Child"},
                    {"field": "Ambiguous Ratio", "headerName": "Ambiguous Ratio", "filter": "agNumberColumnFilter",
                     'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
                ],
    'meek_rules' : [
                    {"field": "Operation"},
                    {"field": "Source"},
                    {"field": "Target"},
                    {"field": "Applicable rule"},
                ]
}

session_objects = dict()

def create_bn():
    session_dict = session_objects[session.get('session_id')]
    user_df = session_dict['user_df']
    categorical_columns = len(
        user_df.select_dtypes(include='category').columns)
    node_names = user_df.columns.to_list()
    node_types = [(x_name, pbn.DiscreteFactorType(
    )) if x.dtype == 'category' else (x_name, pbn.LinearGaussianCPDType()) for x_name, x in user_df.items()]

    if categorical_columns == len(user_df.columns):
        bn = pbn.DiscreteBN(
            nodes=node_names)
    elif categorical_columns == 0 and not session_dict['allow_kde']:
        bn = pbn.GaussianNetwork(
            nodes=node_names)
    elif not session_dict['allow_kde']:
        bn = pbn.CLGNetwork(nodes=node_names, node_types=node_types)
    else:
        bn = pbn.SemiparametricBN(nodes=node_names, node_types=node_types)
        session_dict['selected_score'] = 'CVLikelihood'

    return bn
def setup_score_and_search(whitelist):

    session_dict = session_objects[session.get('session_id')]
    user_df = session_dict['user_df']

    # Preload all absent arcs/edges as the initial blacklist for the Wrapper phase
    if session_dict['hybrid_learning']:
        rows_source = pd.DataFrame(itertools.permutations(user_df.columns, 2), columns=[
            'Source', 'Target'])

        for _, row in rows_source.iterrows():
            source = row['Source']
            target = row['Target']
            if not session_dict['pdag'].has_edge(source, target) \
                    and not session_dict['pdag'].has_arc(target, source) \
                    and [target, source] not in session_dict['node_children_blacklist']:
                session_dict['arc_blacklist'].append(
                    {'Source': target, 'Target': source})

        blacklist_df = pd.DataFrame().from_records(
            session_dict['arc_blacklist'])
        blacklist_df = blacklist_df.drop_duplicates()
        session_dict['arc_blacklist'] = blacklist_df.to_dict('records')

    bn = create_bn()

    # Preload whitelist if in Hybrid Learning
    for source, target in whitelist:
        if [target, source] not in whitelist:
            op = pbn.AddArc(source, target, 0)
            op.apply(bn)
        elif {'Source': target, 'Target': source} in session_dict['arc_whitelist']:
            session_dict['arc_whitelist'].remove(
                {'Source': source, 'Target': target})
            session_dict['arc_whitelist'].remove(
                {'Source': target, 'Target': source})

    bn.fit(user_df)
    session_dict['bn'] = bn
    session_dict['logl'] = [bn.slogl(user_df)]
    session_dict['alpha'] = np.sqrt(np.finfo(float).eps)
    session_dict['limited_indegree'] = True
    session_dict['max_parents'] = 10
    score = create_score(session_dict['selected_score'])
    create_op_set(score)


def setup_contraint_based(blacklist, whitelist):
    session_dict = session_objects[session.get('session_id')]
    user_df = session_dict['user_df']
    pdag = pbn.PartiallyDirectedGraph().CompleteUndirected(
        nodes=user_df.columns.to_list())
    categorical_columns = len(
        user_df.select_dtypes(include='category').columns)
    
    if categorical_columns == len(user_df.columns) or categorical_columns == 0:
        session_dict['selected_i_test'] = 'Mutual Information'
    # Enforce children restrictions on the pdag, no discrete node can have a continuous parent
    elif categorical_columns > 0:
        for source in user_df.columns:
            for target in user_df.columns:
                if source != target and user_df[target].dtype == 'category' and user_df[source].dtype == 'float64':
                    if [source, target] not in session_dict['node_children_blacklist']:
                        session_dict['node_children_blacklist'].append(
                            [source, target])
                    pdag.direct(target, source)

    # Preload constraints if coming back from the Wrapper phase
    for source, target in blacklist:
        if pdag.has_edge(source, target):
            pdag.direct(target, source)
        elif pdag.has_arc(source, target):
            pdag.remove_arc(source, target)
    for source, target in whitelist:
        if pdag.has_edge(source, target):
            pdag.direct(source, target)

    session_dict['pdag'] = pdag
    create_i_test(session_dict['selected_i_test'])
    session_dict['ambiguous_threshold'] = 0.5
    session_dict['alpha'] = 0.05
    session_dict['sepset_size'] = 0
    session_dict['i_test_cache'] = dict()
    session_dict['v_structure_cache'] = list()
    session_dict['bn'] = create_bn()
    
    


def setup_alg():
    session_dict = session_objects[session.get('session_id')]

    # Hybrid learning keeps the constraints between phases
    if not session_dict['hybrid_learning']:
        session_dict['arc_blacklist'] = []
        session_dict['arc_whitelist'] = []
        session_dict['node_children_blacklist'] = list()

    blacklist = [list(d.values()) for d in session_dict['arc_blacklist']]
    whitelist = [list(d.values()) for d in session_dict['arc_whitelist']]

    session_dict['pc_phase'] = 'Adjacency search'

    if session_dict['learning_alg'] == 'score_and_search':
        setup_score_and_search(whitelist)

    else:
        setup_contraint_based(blacklist, whitelist)

    
    session_dict['operator_df'] = build_operator_df()

    session_dict['positive'] = True


def create_i_test(value, n_uncond=5, n_cond=100, k_neigh=10, k_perm=10, samples=10):
    session_dict = session_objects[session.get('session_id')]
    user_df = session_dict['user_df']
    categorical_columns = len(
        user_df.select_dtypes(include='category').columns)
    
    
    match value:
        case 'Mutual Information':
            session_dict['i_test'] = pbn.MutualInformation(
                df=user_df)
        case 'LinearCorr (Cont)':
            if categorical_columns == len(user_df.columns):
                raise ValueError("DataFrame does not contain enough continuous columns.")
            session_dict['i_test'] = pbn.LinearCorrelation(
                df=user_df)
        case 'RCoT (Cont)':
            if categorical_columns == len(user_df.columns):
                raise ValueError("DataFrame does not contain enough continuous columns.")
            session_dict['i_test'] = pbn.RCoT(
                df=user_df, random_fourier_xy=n_uncond, random_fourier_z=n_cond)
        case 'χ2 (Discr)':
            session_dict['i_test'] = pbn.ChiSquare(df=user_df)
        case 'MixedKnnCMI':
            k_neigh = max(len(user_df)//100, k_neigh)
            k_perm = max(len(user_df)//100, k_perm)
            session_dict['i_test'] = pbn.MixedKMutualInformation(df=user_df, k=k_neigh, shuffle_neighbors=k_perm, samples=samples, scaling="min_max", gamma_approx=True, adaptive_k=True)

    return session_dict['i_test']


def create_session(df, filename):
    session_id = session.get('session_id', False)
    if session_id not in session_objects:
        # If session ID doesn't exist, create a new one
        if not session_id:
            session_id = str(uuid4())
            session['session_id'] = session_id
        session_objects[session_id] = dict()
    session_objects[session_id]['temp_df'] = df
    session_objects[session_id]['temp_filename'] = filename.split('.')[0]


def create_op_set(score):
    session_dict = session_objects[session.get('session_id')]

    bn = session_dict['bn']
    op_set = pbn.OperatorPool(opsets=([pbn.ArcOperatorSet()] if not session_dict['allow_kde'] else [
                              pbn.ArcOperatorSet(), pbn.ChangeNodeTypeSet()]))
    op_set.set_arc_blacklist([
        list(d.values()) for d in session_dict['arc_blacklist']]
    )
    op_set.set_arc_whitelist([
        list(d.values()) for d in session_dict['arc_whitelist']]
    )

    op_set.cache_scores(bn, score)

    session_dict['op_set'] = op_set
    return op_set


def create_score(value, kfolds=5):
    session_dict = session_objects[session.get('session_id')]
    user_df = session_dict['user_df']
    match value:
        case 'BIC':
            if str(session_dict['bn'].type()) == 'SemiparametricNetworkType':
                raise ValueError(f"The BN type {
                                 session_dict['bn'].type()} is not compatible with score {value}")
            session_dict['score'] = pbn.BIC(user_df)
        case 'CVLikelihood':
            session_dict['score'] = pbn.CVLikelihood(
                user_df, k=kfolds)
        case 'BDe/BGe (Homog)':

            if str(session_dict['bn'].type()) == 'DiscreteNetworkType':
                session_dict['score'] = pbn.BDe(user_df)
            elif str(session_dict['bn'].type()) == 'GaussianNetworkType':
                session_dict['score'] = pbn.BGe(user_df)
            else:
                raise ValueError(f"The BN type {
                                 session_dict['bn'].type()} is not compatible with score {value}")

    return session_dict['score']

def get_topological_sort_node_types():
    session_dict = session_objects[session.get('session_id')]
    bn = session_dict['bn']
    pdag = session_dict['pdag']
    user_df = session_dict['user_df']

    dag = pdag.to_approximate_dag()
    top_sort = dag.topological_sort()
    for col in user_df:
        if user_df[col].dtype == 'category':
            top_sort.remove(col)

    node_types = [(x_name, bn.node_type(x_name)) for x_name in bn.nodes()]
    spbn =  pbn.SemiparametricBN(nodes=bn.nodes(), node_types=node_types)
    for n1, n2 in dag.arcs():
        spbn.add_arc(n1, n2)

    op_set = pbn.ChangeNodeTypeSet()
    op_set.cache_scores(spbn, session_dict['score'])

    delta = op_set.get_delta()
    type_changes_ops_idx = np.nonzero([len(x) for x in delta])[0]
    type_changes_ops = [['ChangeNodeType', spbn.nodes()[i], ('CKDEType' if str(spbn.node_type(spbn.nodes(
    )[i])) == 'LinearGaussianFactor' else 'LinearGaussianCPDType'), delta[i][0]] for i in type_changes_ops_idx]

    for c in type_changes_ops:
        if c[1] in top_sort and c[3] < 0:
            top_sort.remove(c[1])

    lglk_checks = []
    for n in top_sort:
        check = True
        for p in dag.parents(n):
            if p in top_sort:
                check = False
        if check:
            lglk_checks.append(n)

    type_changes_ops = [x for x in type_changes_ops if x[1] in lglk_checks]

    return type_changes_ops

    
    
def build_operator_entry_score_and_search(idx, bn, delta_raw_df):
    session_dict = session_objects[session.get('session_id')]

    source_idx = idx % len(delta_raw_df)
    target_idx = idx//len(delta_raw_df)

    source = bn.collapsed_name(source_idx)
    target = bn.collapsed_name(target_idx)

    if bn.has_arc(source, target):
        op = pbn.RemoveArc(
            source, target, delta_raw_df.iloc[source_idx, target_idx])
    elif bn.has_arc(target, source) and bn.can_flip_arc(target, source):
        if session_dict['limited_indegree'] and bn.num_parents(target) >= session_dict['max_parents']:
            return [np.nan] * 4
        op = pbn.FlipArc(
            target, source, delta_raw_df.iloc[source_idx, target_idx])
    elif bn.can_add_arc(source, target):
        if session_dict['limited_indegree'] and bn.num_parents(target) >= session_dict['max_parents']:
            return [np.nan] * 4
        op = pbn.AddArc(
            source, target, delta_raw_df.iloc[source_idx, target_idx])
    else:
        return [np.nan] * 4

    return [type(op).__name__, op.source(), op.target(), op.delta()]

def build_operator_df_score_and_search():
    session_dict = session_objects[session.get('session_id')]
    bn = session_dict['bn']
    op_set = session_dict['op_set']
    delta_raw_df = pd.DataFrame(
        op_set.get_op_sets()[0].get_delta(), columns=bn.nodes(), index=bn.nodes())
    # Build the list of delta entries
    delta_operations = [
        build_operator_entry_score_and_search(idx, bn, delta_raw_df)
        for idx in range(delta_raw_df.size)
    ]

    if session_dict['allow_kde']:
        delta = op_set.get_op_sets()[1].get_delta()
        type_changes_ops_idx = np.nonzero([len(x) for x in delta])[0]
        type_changes_ops = [['ChangeNodeType', bn.nodes()[i], ('CKDEType' if str(bn.node_type(bn.nodes(
        )[i])) == 'LinearGaussianFactor' else 'LinearGaussianCPDType'), delta[i][0]] for i in type_changes_ops_idx]
        delta_operations += type_changes_ops

    operator_df = pd.DataFrame(
        delta_operations, columns=session_dict['table_colnames'])
    return operator_df

def build_operator_df_constraint_based():
    session_dict = session_objects[session.get('session_id')]
    ascending_sort = False
    pdag = session_dict['pdag']
    blacklist = [list(d.values()) for d in session_dict['arc_blacklist']]
    whitelist = [list(d.values()) for d in session_dict['arc_whitelist']]
    undirected = list()
    operations = list()
    # Undirect for C++ but keep blacklist, then restore below
    if session_dict['pc_phase'] != 'Meek rules' and session_dict['pc_phase'] != 'PDAG to DAG extension' and session_dict['pc_phase'] != 'Node type selection':
        for source, target in blacklist + session_dict['node_children_blacklist']:
            if pdag.has_arc(target, source):
                pdag.undirect(target, source)
                undirected.append([target, source])
    nodes = session_dict['i_test'].variable_names()

    if session_dict['pc_phase'] == 'Adjacency search':
        if not session_dict['i_test_cache'].get(session_dict.get('sepset_size', 0), False):
            seplist = pbn.PC().compute_sepsets_of_size(pdag=pdag,
                                                        hypot_test=session_dict['i_test'],
                                                        arc_blacklist=blacklist +
                                                        session_dict['node_children_blacklist'],
                                                        arc_whitelist=whitelist,
                                                        sepset_size=session_dict['sepset_size']).l_sep
            
            # Re-apply blacklist
            for source, target in undirected:
                pdag.direct(source, target)

            for operation in seplist:
                edge, evidence, p_value = operation
                node1 = nodes[edge[0]]
                node2 = nodes[edge[1]]
                evidence_nodes = (','.join([nodes[evi]
                                            for evi in evidence]) if evidence else '-')
                # Suggest arc instead of edge if possible
                if pdag.has_arc(node2, node1):
                    node1, node2 = node2, node1

                operations.append(
                    ['RemoveEdge/Arc', node1, node2, evidence_nodes, p_value])
                
            session_dict['i_test_cache'][session_dict.get('sepset_size', 0)] = operations

        # use cached i-tests
        else:
            # Re-apply blacklist
            for source, target in undirected:
                pdag.direct(source, target)

            operations = session_dict['i_test_cache'][session_dict['sepset_size']]
        
    elif session_dict['pc_phase'] == 'V-structure determination':

        ascending_sort = True
        if len(session_dict['v_structure_cache']) == 0:
            vstructures = pbn.PC().compute_v_structures(pdag,
                                                        hypot_test=session_dict['i_test'],
                                                        alpha=session_dict['alpha'],
                                                        use_sepsets=False,
                                                        ambiguous_threshold=1.0)
            # Re-apply blacklist
            for source, target in undirected:
                pdag.direct(source, target)

            for vs in vstructures:
                p1 = nodes[vs.p1]
                p2 = nodes[vs.p2]
                child = nodes[vs.children]
                parent_nodes = ','.join([p1, p2])
                show_operation = True
                # Forbid orientation of blacklisted v_structure edges
                for source, target in blacklist + session_dict['node_children_blacklist']:
                    if (source == p1 and target == child) or (source == p2 and target == child):
                        show_operation = False
                for source, target in whitelist:
                    if (source == child and target == p1) or (source == child and target == p2):
                        show_operation = False

                if pdag.has_arc(p1, child) and pdag.has_arc(p2, child):
                    show_operation = False

                if show_operation:
                    operations.append(
                        ['Create V-structure', parent_nodes, child, vs.ratio])
                    
            session_dict['v_structure_cache'] = operations

        else:
            # Re-apply blacklist
            for source, target in undirected:
                pdag.direct(source, target)

            operations = session_dict['v_structure_cache']
                
    elif session_dict['pc_phase'] == 'Meek rules':
        ascending_sort = True

        applicable_rules = pbn.MeekRules.all_rules_sequential_interactive(
            pdag)

        for rule in applicable_rules:
            arc, rule_number = rule
            source = nodes[arc[0]]
            target = nodes[arc[1]]
            show_operation = not pdag.has_arc(source, target)

            for source_aux, target_aux in blacklist + session_dict['node_children_blacklist']:
                if (source == source_aux and target == target_aux):
                    show_operation = False
            for source, target in whitelist:
                if (source == target_aux and target == source_aux):
                    show_operation = False
            if show_operation:
                operations.append(
                    ['OrientEdge', source, target, rule_number])
                
    elif session_dict['pc_phase'] == 'Node type selection':
        operations.extend(get_topological_sort_node_types())

    operator_df = pd.DataFrame(
        operations, columns=session_dict['table_colnames'])
    
    return operator_df, ascending_sort
def build_operator_df():
    session_dict = session_objects[session.get('session_id')]
    ascending_sort = False
    if session_dict['learning_alg'] == 'score_and_search':
        operator_df = build_operator_df_score_and_search()

    else:
        operator_df, ascending_sort = build_operator_df_constraint_based()
        

    operator_df = operator_df.dropna(axis=0)
    operator_df = operator_df.drop_duplicates()

    operator_df = operator_df.sort_values(
        by=session_dict['table_colnames'][-1], ascending=ascending_sort)

    return operator_df


def build_cytoscape_network():
    session_dict = session_objects[session.get('session_id')]
    G = nx.DiGraph()
    arcs_bidirected = list()
    if session_dict['learning_alg'] == 'score_and_search':
        pdag = session_dict['bn']
        G.add_edges_from(pdag.arcs())
    else:
        pdag = session_dict['pdag']
        arcs_bidirected = pdag.edges()
        arcs = arcs_bidirected + pdag.arcs()
        G.add_edges_from(arcs)

    SCALING_FACTOR = np.log(len(pdag.nodes())) * \
        100  # try using to scatter nodes

    G.add_nodes_from(pdag.nodes())

    # 2) apply a NetworkX layouting algorithm
    layout = session_dict['layout']
    if layout != 'Graphviz neato':
        pos = layout_dict[layout](G)
    else:
        pos = nx.nx_agraph.graphviz_layout(G)
        x_coords, y_coords = zip(*pos.values())
        # Compute min and max values for x and y coordinates
        x_min, x_max = min(x_coords), max(x_coords)
        y_min, y_max = min(y_coords), max(y_coords)
        # Normalize the positions
        normalized_pos = {
            node: (
                2 * (x - x_min) / (x_max - x_min) - 1 if x_max != x_min else 0,
                2 * (y - y_min) / (y_max - y_min) - 1 if y_max != y_min else 0
            )
            for node, (x, y) in pos.items()
        }
        pos = normalized_pos

    # 3) convert networkx graph to cytoscape format
    cy = nx.cytoscape_data(G)

    # 4.) Add the dictionary key label to the nodes list of cy
    for n in cy["elements"]["nodes"]:
        for k, v in n.items():
            v["label"] = v.pop("value")
            
            v["factor_type"] = str(
                session_dict['bn'].node_type(v["label"]))

    # 5.) Add the coords you got from (2) as coordinates of nodes in cy
    for n, p in zip(cy["elements"]["nodes"], pos.values()):
        n["position"] = {"x": int(p[0] * SCALING_FACTOR),
                         "y": int(p[1] * SCALING_FACTOR)}

    # 6.) Add the directed property to the nodes list of cy
    for e in cy["elements"]["edges"]:
        e['data']['directed'] = "True"
        for arc in arcs_bidirected:
            if e['data']['source'] == arc[0] and e['data']['target'] == arc[1]:
                e['data']['directed'] = "False"

    # 7.) Take the results of (3)-(6) and write them to a list, like elements_ls
    elements = cy["elements"]["nodes"] + cy["elements"]["edges"]
    session_dict['cytoscape_elements'] = elements


def update_cytoscape_network(current_elements):
    session_dict = session_objects[session.get('session_id')]
    G = nx.DiGraph()
    arcs_bidirected = list()
    if session_dict['learning_alg'] == 'score_and_search':
        pdag = session_dict['bn']

        G.add_edges_from(pdag.arcs())
    else:
        pdag = session_dict['pdag']
        arcs_bidirected = pdag.edges()
        arcs = arcs_bidirected + pdag.arcs()
        G.add_edges_from(arcs)

    # 1) generate a networkx graph
    G.add_nodes_from(pdag.nodes())

    nodes_new = sorted([x for x in session_dict['cytoscape_elements'] if 'position' in x],
                       key=lambda x: x['data']['id'])
    nodes_old = sorted([x for x in current_elements if 'position' in x],
                       key=lambda x: x['data']['id'])

    # Keep position but update everything else
    for ele_old, ele_new in zip(nodes_old, nodes_new):
        ele_new['position'] = ele_old['position']

    # 2) convert networkx graph to cytoscape format
    cy = nx.cytoscape_data(G)

    for e in cy["elements"]["edges"]:
        e['data']['directed'] = "True"
        for arc in arcs_bidirected:
            if e['data']['source'] == arc[0] and e['data']['target'] == arc[1]:
                e['data']['directed'] = "False"

    # 3.) Modify the edge list
    elements = nodes_new + cy["elements"]["edges"]
    session_dict['cytoscape_elements'] = elements


def filterDf(df, data, col):
    if data["filterType"] == "date":
        crit1 = data["dateFrom"]
        crit1 = pd.Series(crit1).astype(df[col].dtype)[0]
        if "dateTo" in data:
            crit2 = data["dateTo"]
            crit2 = pd.Series(crit2).astype(df[col].dtype)[0]
    else:
        crit1 = data["filter"]
        crit1 = pd.Series(crit1).astype(df[col].dtype)[0]
        if "filterTo" in data:
            crit2 = data["filterTo"]
            crit2 = pd.Series(crit2).astype(df[col].dtype)[0]
    if data["type"] == "contains":
        df = df.loc[df[col].str.contains(crit1)]
    elif data["type"] == "notContains":
        df = df.loc[~df[col].str.contains(crit1)]
    elif data["type"] == "startsWith":
        df = df.loc[df[col].str.startswith(crit1)]
    elif data["type"] == "notStartsWith":
        df = df.loc[~df[col].str.startswith(crit1)]
    elif data["type"] == "endsWith":
        df = df.loc[df[col].str.endswith(crit1)]
    elif data["type"] == "notEndsWith":
        df = df.loc[~df[col].str.endswith(crit1)]
    elif data["type"] == "inRange":
        if data["filterType"] == "date":
            df = df.loc[df[col].astype(
                "datetime64[ns]").between_time(crit1, crit2)]
        else:
            df = df.loc[df[col].between(crit1, crit2)]
    elif data["type"] == "blank":
        df = df.loc[df[col].isnull()]
    elif data["type"] == "notBlank":
        df = df.loc[~df[col].isnull()]
    else:
        df = df.loc[getattr(df[col], operators[data["type"]])(crit1)]
    return df


def apply_rows_request(df, request):
    if request["filterModel"]:
        fils = request["filterModel"]
        for k in fils:
            try:
                if "operator" in fils[k]:
                    if fils[k]["operator"] == "AND":
                        df = filterDf(df, fils[k]["condition1"], k)
                        df = filterDf(df, fils[k]["condition2"], k)
                    else:
                        df1 = filterDf(df, fils[k]["condition1"], k)
                        df2 = filterDf(df, fils[k]["condition2"], k)
                        df = pd.concat([df1, df2])
                else:
                    df = filterDf(df, fils[k], k)
            except:
                pass

    if request["sortModel"]:
        sorting = []
        asc = []
        for sort in request["sortModel"]:
            sorting.append(sort["colId"])
            if sort["sort"] == "asc":
                asc.append(True)
            else:
                asc.append(False)
        df = df.sort_values(by=sorting, ascending=asc)

    df = df.iloc[request["startRow"]: request["endRow"]]
    return df


@app.callback(
    Output("infinite-sort-filter-grid-2", "getRowsResponse"),
    Output("infinite-sort-filter-grid-2", "selectedRows"),
    Output('bt-sepset-increase', 'n_clicks'),

    Input("infinite-sort-filter-grid-2", "getRowsRequest"),
    State("bt-sepset-increase", "disabled"),
    Input("infinite-sort-filter-grid-2", "rowData"),
    prevent_initial_call=True
)
def infinite_scroll(request, sepset_increase_disabled, _):
    session_dict = session_objects.get(session.get('session_id'), None)
    rowdata = [None]
    # Adjacency search increase button
    iterate_phases = True
    
    if session_dict is not None:
        table_colnames = session_dict['table_colnames']
        while iterate_phases:
            rowdata = pd.DataFrame([['-']*len(table_colnames)],
                                   columns=table_colnames).to_dict("records")
            dff = session_dict['operator_df'].copy()

            if request is not None:
                dff = apply_rows_request(dff, request)
                dff = dff.loc[dff[table_colnames
                                  [-1]] > -sys.float_info.max, :]
                if session_dict.get('positive', False) and session_dict['pc_phase'] == 'Adjacency search':
                    dff = dff.loc[dff[table_colnames
                                      [-1]] > session_dict['alpha'], :]
                elif session_dict['pc_phase'] == 'V-structure determination':
                    dff = dff.loc[dff[table_colnames
                                      [-1]] <= session_dict['ambiguous_threshold'], :]

            lines = len(dff.index)

            if lines > 0:
                rowdata = dff.to_dict("records")

            elif session_dict['learning_alg'] == 'constraint_based':
                if not sepset_increase_disabled and session_dict['pc_phase'] == 'Adjacency search' and len(session_dict['operator_df']) > 0 and not request["filterModel"]:
                    session_dict['i_test_cache'].pop(session_dict['sepset_size'], None)
                    session_dict['sepset_size'] += 1
                    operator_df = build_operator_df()
                    session_dict['operator_df'] = operator_df

                    continue

            iterate_phases = False

    return {"rowData": rowdata, "rowCount": len(rowdata)}, [rowdata[0]], -1


def apply_operation_score_and_search(selected_rows, triggered_id):
    session_dict = session_objects[session.get('session_id')]
    bn = session_dict['bn']
    op_set = session_dict['op_set']
    score = session_dict['score']
    render_layout = -1

    if 'btn-row-selection-apply' == triggered_id:

        if selected_rows[0]['Operation'] != 'ChangeNodeType':
            op = getattr(pbn,
                         selected_rows[0]['Operation'])(selected_rows[0]['Source'],
                                                        selected_rows[0]['Target'], 0)
            op.apply(bn)

        else:
            op = pbn.ChangeNodeType(node=selected_rows[0]['Source'], node_type=getattr(
                pbn, selected_rows[0]['Target'])(), delta=0)
            op.apply(bn)
            for ele in session_dict['cytoscape_elements']:
                if ele['data'].get('id', False) and ele['data']['id'] == selected_rows[0]['Source']:
                    ele['data']['factor_type'] = str(
                        bn.node_type(ele['data']['id']))

        op_set.update_scores(bn, score, op.nodes_changed(bn))
        
    # Execute entire algorithm 
    else:
        ghc = pbn.GreedyHillClimbing()
        bn = ghc.estimate(
            start=session_dict['bn'],
            score=score,
            operators=op_set,
            arc_blacklist=[
                list(d.values()) for d in session_dict['arc_blacklist']],
            arc_whitelist=[
                list(d.values()) for d in session_dict['arc_whitelist']],
            max_indegree=session_dict['max_parents'],
            patience=50
        )
        op_set.cache_scores(bn, score)
        if session_dict['allow_kde']:
            for ele in session_dict['cytoscape_elements']:
                if 'position' in ele:
                    ele['data']['factor_type'] = str(
                        bn.node_type(ele['data']['id']))

        render_layout = 0

    total_score = "{:.3f}".format(score.score(bn))

    bn.fit(session_dict['user_df'])
    logl = bn.slogl(session_dict['user_df'])
    session_dict['logl'].append(logl)
    session_dict['bn'] = bn

    graph_data = plotly.graph_objs.Scatter(
        x=np.arange(len(session_dict['logl'])),
        y=session_dict['logl'],
        name='Scatter',
        mode='lines+markers',
    )

    graph_layout = {'data': [graph_data], 'layout': plotly.graph_objs.Layout(xaxis=dict(range=[0, len(session_dict['logl'])-0.5],
                                                                                        title="Steps"),
                                                                             yaxis=dict(range=[min(session_dict['logl']),
                                                                                               max(session_dict['logl']) - max(session_dict['logl'])*0.1],
                                                                                        title=r'$\mathcal{L}\left(\theta,\mathcal{G}\mid\mathcal{D}\right)$'),
                                                                             title='Log-likelihood of the structure'
                                                                             )}
    return total_score, render_layout, graph_layout


def apply_operation_constraint_based(selected_rows, triggered_id):
    session_dict = session_objects[session.get('session_id')]

    pdag = session_dict['pdag']
    bn = session_dict['bn']
    render_layout = -1

    if 'btn-row-selection-apply' == triggered_id and session_dict['pc_phase'] == 'Adjacency search':

        if pdag.has_edge(selected_rows[0]['Node1'], selected_rows[0]['Node2']):
            pdag.remove_edge(
                selected_rows[0]['Node1'], selected_rows[0]['Node2'])
        else:
            pdag.remove_arc(selected_rows[0]['Node1'],
                            selected_rows[0]['Node2'])
            
        operations = session_dict['i_test_cache'][session_dict['sepset_size']]
        session_dict['i_test_cache'][session_dict['sepset_size']] = [x for x in operations if not (x[1] == selected_rows[0]['Node1'] and x[2] == selected_rows[0]['Node2'])]

    

    elif 'btn-row-selection-apply' == triggered_id and session_dict['pc_phase'] == 'V-structure determination':
        p1, p2 = selected_rows[0]['Parents'].split(',')
        pdag.direct(p1, selected_rows[0]['Child'])
        pdag.direct(p2, selected_rows[0]['Child'])


        operations = session_dict['v_structure_cache']
        filtered_ops = list()
        for op in operations:
            if op[1] == selected_rows[0]['Parents'] and op[2] == selected_rows[0]['Child']:
                continue
            p1_aux, p2_aux = op[1].split(',')
            if (p1_aux == selected_rows[0]['Child'] or p2_aux == selected_rows[0]['Child']) and (op[2] == p1 or op[2] == p2):
                continue
            filtered_ops.append(op)
        session_dict['v_structure_cache'] = filtered_ops
      

    elif 'btn-row-selection-apply' == triggered_id and session_dict['pc_phase'] == 'Meek rules':
        pdag.direct(selected_rows[0]['Source'], selected_rows[0]['Target'])

    elif 'btn-row-selection-apply' == triggered_id and session_dict['pc_phase'] == 'Node type selection':
        op = pbn.ChangeNodeType(node=selected_rows[0]['Node'], node_type=getattr(
                pbn, selected_rows[0]['Type'])(), delta=0)
        op.apply(bn)
        for ele in session_dict['cytoscape_elements']:
            if ele['data'].get('id', False) and ele['data']['id'] == selected_rows[0]['Node']:
                ele['data']['factor_type'] = str(
                    bn.node_type(ele['data']['id']))
    

    else:
        # IN CASE ALL REMAINING PHASES
        blacklist = [list(d.values()) for d in session_dict['arc_blacklist']]
        whitelist = [list(d.values()) for d in session_dict['arc_whitelist']]
        undirected_arcs = list()
        # Undirect for C++ but keep blacklist, then restore below
        for source, target in blacklist + session_dict['node_children_blacklist']:
            if pdag.has_arc(target, source):
                pdag.undirect(target, source)
                undirected_arcs.append([target, source])

        if session_dict['hybrid_learning']:
            pdag = pbn.PC().apply_adjacency_search(pdag=pdag,
                                                   hypot_test=session_dict['i_test'],
                                                   arc_blacklist=blacklist +
                                                   session_dict['node_children_blacklist'],
                                                   arc_whitelist=whitelist,
                                                   alpha=session_dict['alpha'],
                                                   )
            # Re-apply blacklist
            for arc in undirected_arcs:
                if pdag.has_edge(arc[0], arc[1]):
                    pdag.direct(arc[0], arc[1])

        else:
            match session_dict['pc_phase']:
                case 'Adjacency search':
                    phase_number = 0
                case 'V-structure determination':
                    phase_number = 1
                case 'Meek rules':
                    phase_number = 2

            # Apply remaining phases
            new_pdag = pbn.PC().estimate_from_initial_pdag(pdag=pdag,
                                                           hypot_test=session_dict['i_test'],
                                                           arc_blacklist=blacklist +
                                                           session_dict['node_children_blacklist'],
                                                           arc_whitelist=whitelist,
                                                           alpha=session_dict['alpha'],
                                                           ambiguous_threshold=session_dict['ambiguous_threshold'],
                                                           phase_number=phase_number)
            
            # Re-apply blacklist
            for arc in undirected_arcs:
                if new_pdag.has_edge(arc[0], arc[1]):
                    new_pdag.direct(arc[0], arc[1])
            
            # Try to extend to DAG, if not possible apply topological ordering method
            try:
                dag = new_pdag.to_dag()
            except ValueError:
                dag = new_pdag.to_approximate_dag()

            for arc in dag.arcs():
                if pdag.has_edge(arc[0], arc[1]):
                    pdag.direct(arc[0], arc[1])
                elif pdag.has_arc(arc[1], arc[0]):
                    pdag.flip_arc(arc[1], arc[0])
                elif not pdag.has_arc(arc[0], arc[1]):
                    pdag.add_arc(arc[0], arc[1])


        session_dict['i_test_cache'] = dict()
        session_dict['v_structure_cache'] = list()      

        render_layout = 0

    return render_layout


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output('total-score', 'children', allow_duplicate=True),
    Output('bt-reset', 'n_clicks', allow_duplicate=True),
    Output('cytoscape', 'tapEdgeData'),
    Output('logl-graph', 'figure', allow_duplicate=True),
    Output('next-pc-button', 'n_clicks'),
    State("infinite-sort-filter-grid-2", "selectedRows"),

    Input("btn-row-selection-apply", "n_clicks"),
    Input("btn-call-library", "n_clicks"),
    prevent_initial_call=True
)
def apply_selected_operation(selected_rows, *_):
    session_dict = session_objects[session.get('session_id')]

    triggered_id = ctx.triggered_id
    total_score = graph_layout = render_layout = next_pc_phase = no_update

    if selected_rows and ('btn-row-selection-apply' == triggered_id or 'btn-call-library' == triggered_id) and not session_dict.get('operations_loading', False):
        if selected_rows[0]['Operation'] == '-':
             # Click next pc phase if current is finished
            if (session_dict['pc_phase'] != 'PDAG to DAG extension' and session_dict['learning_alg'] == 'constraint_based')\
                    or (session_dict['pc_phase'] == 'Adjacency search' and session_dict['learning_alg'] == 'constraint_based' and session_dict['hybrid_learning']):
                next_pc_phase = 1

            return no_update, no_update, no_update, no_update, no_update, next_pc_phase
        
        # Trigger loading component after 1 sec, also avoids call stacking
        session_dict['operations_loading'] = True

        if session_dict['learning_alg'] == 'score_and_search':
            total_score, render_layout, graph_layout = apply_operation_score_and_search(
                selected_rows, triggered_id)

        else:
            render_layout = apply_operation_constraint_based(
                selected_rows, triggered_id)

        session_dict['operator_df'] = build_operator_df()
        session_dict['operations_loading'] = False

    rowdata = session_dict['operator_df'].to_dict('records')

    return rowdata, total_score, render_layout, {'id': session_dict.get('selected_edge', False)}, graph_layout, next_pc_phase


@app.callback(
    Output('cytoscape', 'zoom'),
    Output('cytoscape', 'elements'),
    State('cytoscape', 'elements'),
    Input('bt-reset', 'n_clicks'),
    prevent_initial_call=True

)
def load_layout(current_elements, n_clicks):
    session_dict = session_objects[session.get('session_id')]

    if n_clicks != -1 or n_clicks is None or not current_elements:

        build_cytoscape_network()
        zoom = 1

    else:
        update_cytoscape_network(current_elements)
        zoom = no_update
    elements = session_dict['cytoscape_elements']
    return zoom, elements


@app.callback(
    Output('bt-reset', 'n_clicks', allow_duplicate=True),
    Input('layout-dropdown', 'value'),
    prevent_initial_call=True
)
def change_layout_type(value):
    session_dict = session_objects[session.get('session_id')]
    session_dict['layout'] = value
    return 1


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Input('toggle-positive-operations', 'value'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def filter_positive_entries(value, rowdata, getRowsRequest):
    session_dict = session_objects[session.get('session_id')]
    if value is not None and len(value) > 0:
        session_dict['positive'] = True
    else:
        session_dict['positive'] = False
    return rowdata, getRowsRequest


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Input('input-alpha', 'value'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_alpha_significance_level(value, rowdata, getRowsRequest):
    session_dict = session_objects[session.get('session_id')]
    session_dict['alpha'] = value
    if session_dict['pc_phase'] == 'V-structure determination':
        session_dict['v_structure_cache'] = list()
        session_dict['operator_df'] = build_operator_df()

        rowdata = session_dict['operator_df'].to_dict('records')

    return rowdata, getRowsRequest


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Input('input-ambiguous-threshold', 'value'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_ambiguous_threshold(value, rowdata, getRowsRequest):
    session_dict = session_objects[session.get('session_id')]
    session_dict['ambiguous_threshold'] = value

    return rowdata, getRowsRequest


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Output("pc-phase-text", "children", allow_duplicate=True),
    Output("bt-sepset-increase", "disabled", allow_duplicate=True),
    Input('bt-sepset-increase', 'n_clicks'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_sepset_size(n_clicks, rowData, getRowsRequest):
    session_dict = session_objects.get(session.get('session_id'), None)

    pc_phase_text = next_forbidden = no_update

    if session_dict is not None and session_dict['learning_alg'] == 'constraint_based':
        if session_dict['pc_phase'] == 'Adjacency search':
            if n_clicks != -1:
                session_dict['i_test_cache'].pop(session_dict['sepset_size'], None)
                session_dict['sepset_size'] += 1
                operator_df = build_operator_df()
                session_dict['operator_df'] = operator_df
            else:
                operator_df = session_dict['operator_df']
            pc_phase_text = ["Adjacency search ",
                             fr'$\ \ell = {session_dict['sepset_size']}$']
            next_forbidden = (len(operator_df) == 0)
    return rowData, getRowsRequest, pc_phase_text, next_forbidden


@app.callback(
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Output("pc-phase-text", "children", allow_duplicate=True),
    Output("div-sepset-increase", "hidden", allow_duplicate=True),
    Output("div-ambiguous-threshold", "hidden", allow_duplicate=True),
    Output("input-ambiguous-threshold", "value", allow_duplicate=True),
    Output("next-pc-button", "disabled", allow_duplicate=True),
    Output("prev-pc-button", "disabled", allow_duplicate=True),
    Output("toggle-positive-operations", "options", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "columnDefs", allow_duplicate=True),
    Output("div-orient", "hidden", allow_duplicate=True),
    Output("bt-orient", "disabled", allow_duplicate=True),
    Output('btn-learn-score-and-search', 'n_clicks', allow_duplicate=True),
    Output('btn-learn-constraint-based', 'n_clicks', allow_duplicate=True),
    Input('next-pc-button', 'n_clicks'),
    Input('prev-pc-button', 'n_clicks'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_pc_phase(next, prev, rowData, getRowsRequest):
    session_dict = session_objects.get(session.get('session_id'), None)

    pc_phase_text = next_forbidden = prev_forbidden = sepset_increase_hide = disable_toggle_filter = columnDefs = ambiguous_threshold_hide = ambiguous_threshold_value = hide_orient = disable_orient = change_hybrid_next_phase = change_hybrid_prev_phase = no_update

    triggered_id = ctx.triggered_id
    if session_dict is not None and session_dict['learning_alg'] == 'constraint_based':
        if session_dict['pc_phase'] == 'Adjacency search':
            
            session_dict['i_test_cache'] = dict()
            if not session_dict['hybrid_learning']:
                session_dict['pc_phase'] = 'V-structure determination'
                next_forbidden = prev_forbidden = ambiguous_threshold_hide = False
                session_dict['ambiguous_threshold'] = ambiguous_threshold_value = 0.5
                sepset_increase_hide = True
                disable_toggle_filter = [
                    {'label': 'Only significant operations', 'value': 'option1', 'disabled': True}]
                pc_phase_text = 'V-structure determination'
                columnDefs = column_definitions_dict['v_structure_determination']
                session_dict['table_colnames'] = [x['field'] for x in columnDefs]
            else:
                prev_forbidden = False
                next_forbidden = sepset_increase_hide = True
                pc_phase_text = 'Constrained optimization'
                change_hybrid_next_phase = -1
                getRowsRequest = rowData = no_update

        elif session_dict['pc_phase'] == 'V-structure determination':
            session_dict['v_structure_cache'] = list()
            if triggered_id == 'prev-pc-button':
                session_dict['pc_phase'] = 'Adjacency search'
                prev_forbidden = ambiguous_threshold_hide = True
                session_dict['sepset_size'] = 0
                sepset_increase_hide = False
                disable_toggle_filter = [
                    {'label': 'Only significant operations', 'value': 'option1', 'disabled': False}]
                pc_phase_text = ["Adjacency search ",
                                 fr'$\ \ell = {session_dict['sepset_size']}$']
                columnDefs = column_definitions_dict['adjacency_search']
                session_dict['table_colnames'] = [x['field'] for x in columnDefs]
            else:
                session_dict['pc_phase'] = 'Meek rules'
                next_forbidden = prev_forbidden = False
                ambiguous_threshold_hide = True
                pc_phase_text = 'Meek orientation rules'
                columnDefs = column_definitions_dict['meek_rules']
                
                session_dict['table_colnames'] = ['Operation',
                                                  'Source', 'Target', 'Applicable rule']
        elif session_dict['pc_phase'] == 'Meek rules':
            if triggered_id == 'prev-pc-button':
                session_dict['pc_phase'] = 'V-structure determination'
                next_forbidden = prev_forbidden = ambiguous_threshold_hide = False
                session_dict['ambiguous_threshold'] = ambiguous_threshold_value = 0.5
                pc_phase_text = 'V-structure determination'
                columnDefs = column_definitions_dict['v_structure_determination']
                session_dict['table_colnames'] = [x['field'] for x in columnDefs]
            else:
                session_dict['pc_phase'] = pc_phase_text = 'PDAG to DAG extension'
                next_forbidden = not session_dict['allow_kde']
                hide_orient = False
                disable_orient = len(session_dict['pdag'].edges()) == 0

        elif session_dict['pc_phase'] == 'PDAG to DAG extension':
            if triggered_id == 'prev-pc-button':
                session_dict['pc_phase'] = 'Meek rules'
                next_forbidden = prev_forbidden = False
                hide_orient = True
                pc_phase_text = 'Meek orientation rules'
            else:
                session_dict['pc_phase'] = 'Node type selection'
                prev_forbidden = False
                hide_orient = next_forbidden = True
                pc_phase_text = 'Node type selection (Topological order)'
                disable_toggle_filter = [
                    {'label': 'Only significant operations', 'value': 'option1', 'disabled': False}]
                columnDefs = column_definitions_dict['node_type_selection']
                session_dict['table_colnames'] = [x['field'] for x in columnDefs]
        else:
            session_dict['pc_phase'] = pc_phase_text = 'PDAG to DAG extension'
            next_forbidden = not session_dict['allow_kde']
            hide_orient = False
            disable_orient = len(session_dict['pdag'].edges()) == 0


        operator_df = build_operator_df()
        session_dict['operator_df'] = operator_df

    else:

        prev_forbidden = True
        next_forbidden = sepset_increase_hide = False
        session_dict['sepset_size'] = 0
        pc_phase_text = ["Adjacency search ",
                         fr'$\ \ell = {session_dict['sepset_size']}$']
        columnDefs = column_definitions_dict['adjacency_search']
        session_dict['table_colnames'] = [x['field'] for x in columnDefs]
        change_hybrid_prev_phase = -1
        getRowsRequest = rowData = no_update

    return rowData, getRowsRequest, \
    pc_phase_text, sepset_increase_hide, \
    ambiguous_threshold_hide, \
    ambiguous_threshold_value, \
    next_forbidden, prev_forbidden, \
    disable_toggle_filter, columnDefs, \
    hide_orient, disable_orient, \
    change_hybrid_next_phase, change_hybrid_prev_phase


@app.callback(
    Output("apply-loop", "disabled"),
    Input('start-stop-button', 'on'),
    prevent_initial_call=True
)
def toggle_auto_apply_button(on):
    return not on


@app.callback(
    Output("btn-row-selection-apply", "n_clicks"),
    Output('start-stop-button', 'on'),

    Input('apply-loop', 'n_intervals'),
    State('start-stop-button', 'on'),
    State('btn-row-selection-apply', 'n_clicks'),
    State("infinite-sort-filter-grid-2", "selectedRows"),
    prevent_initial_call=True
)
def update_auto_apply_clicks(_, on, current_clicks, selected_rows):
    session_dict = session_objects[session.get('session_id')]

    if not on:
        return no_update, no_update
    elif current_clicks is None:
        return 1, no_update
    elif session_dict['learning_alg'] == 'score_and_search' and (selected_rows[0]['Operation'] == '-'
                                                                 or (session_dict['positive']
                                                                     and selected_rows[0][session_dict['table_colnames'][-1]] <= session_dict['alpha'])) \
            or session_dict['pc_phase'] == 'PDAG to DAG extension':
        return no_update, False

    return current_clicks + 1, no_update


@app.callback(Output('apply-loop', 'interval'),
              Input('speed-slider', 'value'),
              prevent_initial_call=True)
def update_speed(value):
    return value


@app.callback(Output("infinite-sort-filter-grid-2", "getRowsRequest", allow_duplicate=True),
              Output("infinite-sort-filter-grid-2",
                     "rowData", allow_duplicate=True),
              Output('cytoscape', 'stylesheet', allow_duplicate=True),
              Output('node-factor-notification', 'is_open'),
              Output('node-factor-text', 'children'),
              Input('cytoscape', 'tapNodeData'),
              State("infinite-sort-filter-grid-2", "rowData"),
              prevent_initial_call=True)
def displayTapNodeData(data, rowData):
    if data:
        session_dict = session_objects[session.get('session_id')]
        bn = session_dict['bn']
        notification = notification_text = no_update
        if session_dict.get('selected_node', False) != data['label']:
            colname = 'Source' if session_dict['learning_alg'] == 'score_and_search' else 'Node1'
            filter_dict = {'startRow': 0, 'endRow': 100, 'sortModel': [], 'filterModel': {
                colname: {'filterType': 'text', 'type': 'equals', 'filter': data['label']}}}

            stylesheet = default_stylesheet + [{
                'selector': f'node[id = "{data['label']}"]',
                'style': {
                    'background-color': '#00A2E1',  # color for clicked node
                    'width': 'mapData(size, 0, 100, 10, 20)',
                    'height': 'mapData(size, 0, 100, 10, 20)',
                }
            }]
            session_dict['selected_node'] = data['label']
            notification = True
            node_type = bn.node_type(data['label'])
            match str(node_type):
                case 'DiscreteFactor':
                    notification_text = "Discrete"
                case 'LinearGaussianFactor':
                    notification_text = "Gaussian"
                case 'CKDEFactor':
                    notification_text = "KDE"
        else:
            filter_dict = {'startRow': 0, 'endRow': 100,
                           'sortModel': [], 'filterModel': {}}

            stylesheet = default_stylesheet
            session_dict['selected_node'] = False
        return filter_dict, rowData, stylesheet, notification, notification_text
    else:
        return no_update, no_update, no_update, no_update, no_update


@app.callback(Output("infinite-sort-filter-grid-2", "getRowsRequest", allow_duplicate=True),
              Output("infinite-sort-filter-grid-2",
                     "rowData", allow_duplicate=True),
              Output('cytoscape', 'stylesheet', allow_duplicate=True),
              Input('cytoscape', 'tapEdgeData'),
              State("infinite-sort-filter-grid-2", "rowData"),
              prevent_initial_call=True)
def displayTapEdgeData(data, rowData):
    if data:
        session_dict = session_objects[session.get('session_id')]

        if session_dict.get('selected_edge', False) != data['id']:
            colname1 = 'Source' if session_dict['learning_alg'] == 'score_and_search' else 'Node1'
            colname2 = 'Target' if session_dict['learning_alg'] == 'score_and_search' else 'Node2'
            filter_dict = {'startRow': 0, 'endRow': 100, 'sortModel': [], 'filterModel':
                           {colname1: {'filterType': 'text', 'type': 'equals', 'filter': data['source']},
                            colname2: {'filterType': 'text', 'type': 'equals', 'filter': data['target']}}}

            pdag = session_dict['pdag']
            arcs_bidirected = pdag.edges()
            directed = True
            for arc in arcs_bidirected:
                if data['source'] == arc[0] and data['target'] == arc[1]:
                    directed = False
            stylesheet = default_stylesheet + [{
                'selector': f'edge[id = "{data['id']}"]',
                'style': {
                    'line-color': '#00A2E1',  # color for clicked edge
                    'width': 2,
                    'curve-style': 'bezier',
                    'target-arrow-shape': 'triangle' if directed else 'none',
                    'target-arrow-color': '#007bff',  # Arrowhead color
                    'source-arrow-color': '#007bff', }
            }]

            session_dict['selected_edge'] = data['id']

        else:
            filter_dict = {'startRow': 0, 'endRow': 100,
                           'sortModel': [], 'filterModel': {}}

            stylesheet = default_stylesheet
            session_dict['selected_edge'] = False
        return filter_dict, rowData, stylesheet
    else:

        no_update, no_update, no_update


@app.callback(
    Output('whitelisted-arcs-grid', 'rowData'),
    Output('blacklisted-arcs-grid', 'rowData', allow_duplicate=True),
    Output('select-arcs-grid', 'rowData'),
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output('bt-reset', 'n_clicks', allow_duplicate=True),
    Output('total-score', 'children', allow_duplicate=True),
    Output("notification", "is_open", allow_duplicate=True),
    Output("notification-text", "children", allow_duplicate=True),
    Output('logl-graph', 'figure', allow_duplicate=True),
    Input('add-whitelist-button', 'n_clicks'),
    Input('add-blacklist-button', 'n_clicks'),
    State('select-arcs-grid', 'selectedRows'),
    prevent_initial_call=True,

)
def apply_add_constraint(add_whitelist, add_blacklist, selected_rows_source):
    session_dict = session_objects[session.get('session_id')]
    bn = session_dict['bn']

    operation_table_data = update_layout = total_score = rows_source = notification = notification_text = figure_data = no_update
    triggered_id = ctx.triggered_id

    # Populate source table at startup
    if add_whitelist == 0:
        rows_source = pd.DataFrame(itertools.permutations(bn.nodes(), 2), columns=[
                                   'Source', 'Target'])
        for idx, row in rows_source.iterrows():
            if [row['Source'], row['Target']] in session_dict['node_children_blacklist']:
                rows_source = rows_source.drop(index=idx)
        rows_source = rows_source.reset_index(drop=True).to_dict("records")

    try:
        if add_whitelist > 0 and triggered_id == 'add-whitelist-button' and selected_rows_source:
            selected_rows_check = session_dict['arc_whitelist'] + [
                x for x in selected_rows_source if x not in session_dict['arc_whitelist']]
            operation_table_data, update_layout, total_score = check_constraints(
                'arc_whitelist', 'arc_blacklist', selected_rows_check, pbn.AddArc, selected_rows_source)

        if add_blacklist > 0 and triggered_id == 'add-blacklist-button' and selected_rows_source:
            selected_rows_check = session_dict['arc_blacklist'] + [
                x for x in selected_rows_source if x not in session_dict['arc_blacklist']]

            operation_table_data, update_layout, total_score = check_constraints(
                'arc_blacklist', 'arc_whitelist', selected_rows_check, pbn.RemoveArc, selected_rows_source)

        session_dict['bn'].fit(session_dict['user_df'])
        logl = session_dict['bn'].slogl(session_dict['user_df'])
        session_dict['logl'].append(logl)

        graph_data = plotly.graph_objs.Scatter(
            x=np.arange(len(session_dict['logl'])),
            y=session_dict['logl'],
            name='Scatter',
            mode='lines+markers',


        )

        figure_data = {'data': [graph_data], 'layout': plotly.graph_objs.Layout(xaxis=dict(range=[0, len(session_dict['logl'])-0.5],
                                                                                           title="Steps"),
                                                                                yaxis=dict(range=[min(session_dict['logl']),
                                                                                                  max(session_dict['logl']) - max(session_dict['logl'])*0.1],
                                                                                           title=r'$\mathcal{L}\left(\theta,\mathcal{G}\mid\mathcal{D}\right)$'),
                                                                                title='Log-likelihood of the structure'
                                                                                )}

    except Exception as e:
        notification = True
        notification_text = str(e)

    return session_dict['arc_whitelist'], \
        session_dict['arc_blacklist'], \
        rows_source, operation_table_data, \
        update_layout, total_score, \
        notification, notification_text, figure_data


def check_constraints(constraint_list, counter_list, check_list, operator, selected_rows_source):
    session_dict = session_objects[session.get('session_id')]
    total_score = no_update
    # Check with the opposite list
    overlaps = [value for value in session_dict[counter_list]
                if value in check_list]
    if overlaps:
        raise Exception(
            f'Arc {overlaps[0]['Source']} -> {overlaps[0]['Target']} in {counter_list[4:]}')

    constraint_list_bn = [
        list(d.values()) for d in check_list]
    if session_dict['learning_alg'] == 'score_and_search':
        bn = session_dict['bn']
        score = session_dict['score']
        op_set = session_dict['op_set']

        if constraint_list == 'arc_whitelist':
            bn_copy = bn.clone()
            # First check with pybnesian and blacklist
            bn_copy.force_whitelist(constraint_list_bn)

        # Confirm changes
        session_dict[constraint_list] = check_list

        op_set.set_arc_blacklist([
            list(d.values()) for d in session_dict['arc_blacklist']]
        )
        op_set.set_arc_whitelist([
            list(d.values()) for d in session_dict['arc_whitelist']]
        )
        for listed in selected_rows_source:
            op = operator(listed['Source'], listed['Target'], 0)
            op.apply(bn)
            op_set.update_scores(
                bn, score, [listed['Source'], listed['Target']])
        # Update and cache scores
        op_set.cache_scores(bn, score)
        total_score = "{:.3f}".format(score.score(bn))

    else:
        pdag = session_dict['pdag']
        for source, target in constraint_list_bn:
            if constraint_list == 'arc_whitelist':
                if pdag.has_arc(target, source):
                    pdag.undirect(target, source)
                elif pdag.has_edge(source, target):
                    pdag.direct(source, target)
                elif not pdag.has_arc(source, target):
                    pdag.add_arc(source, target)
            else:

                if pdag.has_edge(source, target):
                    pdag.remove_edge(source, target)
                    pdag.add_arc(target, source)

                elif pdag.has_arc(source, target):
                    pdag.remove_arc(source, target)
        # Confirm changes
        session_dict[constraint_list] = check_list
        session_dict['i_test_cache'] = dict()
        session_dict['v_structure_cache'] = list()

    operator_df = build_operator_df()
    session_dict['operator_df'] = operator_df

    return session_dict['operator_df'].to_dict('records'), -1, total_score


@app.callback(
    Output('whitelisted-arcs-grid', 'rowData', allow_duplicate=True),
    Output('blacklisted-arcs-grid', 'rowData', allow_duplicate=True),
    Output("infinite-sort-filter-grid-2", "rowData", allow_duplicate=True),
    Output('bt-reset', 'n_clicks', allow_duplicate=True),
    Input('remove-whitelist-button', 'n_clicks'),
    State('whitelisted-arcs-grid', 'selectedRows'),
    Input('remove-blacklist-button', 'n_clicks'),
    State('blacklisted-arcs-grid', 'selectedRows'),
    prevent_initial_call=True
)
def apply_remove_constraint(remove_whitelist, selected_rows_whitelist, remove_blacklist, selected_rows_blacklist):
    session_dict = session_objects[session.get('session_id')]

    operation_table_data = update_layout = no_update

    triggered_id = ctx.triggered_id

    if remove_whitelist is not None and remove_blacklist is not None:
        update_layout = -1
        if session_dict['learning_alg'] == 'score_and_search':
            bn = session_dict['bn']
            score = session_dict['score']
            op_set = session_dict['op_set']
            if triggered_id == 'remove-whitelist-button' and selected_rows_whitelist:
                session_dict['arc_whitelist'] = [
                    d for d in session_dict['arc_whitelist'] if d not in selected_rows_whitelist]

                op_set.set_arc_whitelist([
                    list(d.values()) for d in session_dict['arc_whitelist']])

            if triggered_id == 'remove-blacklist-button' and selected_rows_blacklist:
                session_dict['arc_blacklist'] = [
                    d for d in session_dict['arc_blacklist'] if d not in selected_rows_blacklist]

                op_set.set_arc_blacklist([
                    d.values() for d in session_dict['arc_blacklist']])

            op_set.cache_scores(bn, score)
        else:
            pdag = session_dict['pdag']
            whitelist = [list(d.values())
                         for d in session_dict['arc_whitelist']]
            blacklist = [list(d.values())
                         for d in session_dict['arc_blacklist']]
            if triggered_id == 'remove-whitelist-button' and selected_rows_whitelist:
                new_whitelist = [
                    d for d in session_dict['arc_whitelist'] if d not in selected_rows_whitelist]
                session_dict['arc_whitelist'] = new_whitelist
                for source, target in [list(d.values()) for d in selected_rows_whitelist]:
                    if [target, source] not in blacklist:
                        if pdag.has_edge(source, target) and [target, source] in [list(d.values()) for d in new_whitelist]:
                            pdag.direct(target, source)
                        else:
                            pdag.undirect(source, target)

            if triggered_id == 'remove-blacklist-button' and selected_rows_blacklist:
                new_blacklist = [
                    d for d in session_dict['arc_blacklist'] if d not in selected_rows_blacklist]
                session_dict['arc_blacklist'] = new_blacklist
                for source, target in [list(d.values()) for d in selected_rows_blacklist]:

                    if [target, source] not in whitelist:
                        if pdag.has_arc(target, source):
                            pdag.undirect(target, source)
                        elif [target, source] in [list(d.values()) for d in new_blacklist]:
                            pdag.add_arc(source, target)
                        else:
                            pdag.add_edge(source, target)

            session_dict['i_test_cache'] = dict()
            session_dict['v_structure_cache'] = list()

        session_dict['operator_df'] = build_operator_df()
        operation_table_data = session_dict['operator_df'].to_dict('records')

    return session_dict['arc_whitelist'], session_dict['arc_blacklist'], operation_table_data, update_layout


@app.callback(
    Output("infinite-sort-filter-grid-2",
           "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Output('total-score', 'children', allow_duplicate=True),
    Output("notification", "is_open", allow_duplicate=True),
    Output("notification-text", "children", allow_duplicate=True),
    Output('score-dropdown', 'value', allow_duplicate=True),
    Output('cvlikelihood-hparams', 'hidden', allow_duplicate=True),
    Input('score-dropdown', 'value'),
    Input('k-folds', 'value'),
    Input('max-parents', 'value'),

    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_score_type(selected_score, kfolds, max_parents, rowData, getRowsRequest):
    session_dict = session_objects[session.get('session_id')]
    notification = notification_text = accepted_score = no_update
    try:

        if selected_score is None:
            selected_score = 'CVLikelihood'
        score = create_score(selected_score, kfolds)
        create_op_set(score)
        session_dict['selected_score'] = selected_score
        session_dict['max_parents'] = max_parents
        session_dict['operator_df'] = build_operator_df()
        total_score = "{:.3f}".format(
            score.score(session_dict['bn']))
        return rowData, getRowsRequest, total_score, notification, notification_text, accepted_score, (selected_score != 'CVLikelihood')
    except ValueError as e:
        notification = True
        notification_text = str(e)
        accepted_score = session_dict['selected_score']

    return no_update, no_update, no_update, notification, notification_text, accepted_score, no_update


@app.callback(
    Output('bt-reset', 'n_clicks', allow_duplicate=True),
    Output("bt-orient", "disabled", allow_duplicate=True),
    Input('bt-orient', 'n_clicks'),
    prevent_initial_call=True
)
def apply_pdag_extension(nclicks):
    session_dict = session_objects[session.get('session_id')]
    pdag = session_dict['pdag']
    try:
        dag = pdag.to_dag()
    except ValueError as e:
        dag = pdag.to_approximate_dag()

    for arc in dag.arcs():
        if pdag.has_edge(arc[0], arc[1]):
            pdag.direct(arc[0], arc[1])
        elif pdag.has_arc(arc[1], arc[0]):
            pdag.flip_arc(arc[1], arc[0])

    return -1, True


@app.callback(
    Output("infinite-sort-filter-grid-2",
           "rowData", allow_duplicate=True),
    Output("infinite-sort-filter-grid-2",
           "getRowsRequest", allow_duplicate=True),
    Output("notification", "is_open", allow_duplicate=True),
    Output("notification-text", "children", allow_duplicate=True),
    Output('itest-dropdown', 'value', allow_duplicate=True),
    Output('rcot-hparams', 'hidden', allow_duplicate=True),
    Output('knncmi-hparams', 'hidden', allow_duplicate=True),
    Input('itest-dropdown', 'value'),
    Input('n-uncond', 'value'),
    Input('n-cond', 'value'),
    Input('k-neigh', 'value'),
    Input('k-perm', 'value'),
    Input('samples', 'value'),
    State("infinite-sort-filter-grid-2", "rowData"),
    State("infinite-sort-filter-grid-2",
          "getRowsRequest"),
    prevent_initial_call=True
)
def change_i_test_type(selected_i_test, n_uncond, n_cond, k_neigh, k_perm, samples, rowData, getRowsRequest):
    session_dict = session_objects[session.get('session_id')]
    triggered_id = ctx.triggered_id
    notification = notification_text = accepted_i_test = no_update
    try:
        if 'infinite-sort-filter-grid-2' != triggered_id:
            if selected_i_test is None:
                selected_i_test = 'Mutual Information'
            create_i_test(selected_i_test, n_uncond, n_cond, k_neigh, k_perm, samples)
            session_dict['i_test_cache'] = dict()
            session_dict['v_structure_cache'] = list()
            session_dict['operator_df'] = build_operator_df()
            session_dict['selected_i_test'] = selected_i_test
            return rowData, getRowsRequest, notification, notification_text, accepted_i_test, (selected_i_test != 'RCoT (Cont)'), (selected_i_test != 'MixedKnnCMI')
    except ValueError as e:
        notification = True
        notification_text = str(e)
        accepted_i_test = session_dict['selected_i_test']
        create_i_test(accepted_i_test, n_uncond, n_cond, k_neigh, k_perm, samples,)

    return no_update, no_update, notification, notification_text, accepted_i_test, no_update, no_update


def linear_dependent_features(dataset):
    to_delete = set()
    dataset = dataset.select_dtypes('float64')
    if not dataset.empty:
        rank = np.linalg.matrix_rank(dataset.cov())
        df_copy = dataset.copy()
        if rank < dataset.shape[1]:
            for c in dataset.columns:
                new_df = df_copy.drop(c, axis=1)
                new_rank = np.linalg.matrix_rank(new_df.cov())

                if rank <= new_rank:
                    to_delete.add(c)
                    df_copy = new_df

    return to_delete


def parse_content(content, filename):
    _, content_string = content.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if '.csv' in filename or '.data' in filename or '.dat' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')), sep=None, engine='python', na_values='?')

        df = df.dropna(axis=1, thresh=int(0.3*len(df)))
        df = df.dropna(axis=0)

        index_constant = np.where(df.nunique() == 1)[0]
        constant_columns = [df.columns[i] for i in index_constant]
        df = df.drop(columns=constant_columns, axis=1)

        # Sample if excesively big??

        cat_data = df.select_dtypes('object').astype('category')
        for c in cat_data:
            df = df.assign(**{c: cat_data[c]})

        float_data = df.select_dtypes('number').astype('float64')
        for c in float_data:
            df = df.assign(**{c: float_data[c]})

        df.reset_index(drop=True, inplace=True)

        to_remove_features = linear_dependent_features(df)
        df = df.drop(columns=to_remove_features, axis=1)
        return df
    except Exception:
        return None


def populate_modal(df):

    columns = df.columns
    selection_options = []

    # For each column, create a radio button to choose between "Categorical" and "Continuous"
    for col in columns:
        not_numeric = False
        try:
            df[col].astype('float64')
        except ValueError:
            not_numeric = True

        selection_options.append(
            html.Div(
                [
                    dbc.Label(f"{col}"),
                    dbc.RadioItems(
                        options=[
                            {"label": "Categorical/Discrete", "value": 'category'},
                            {"label": "Continuous", "value": 'float64',
                                "disabled": not_numeric}
                        ],
                        value=str(df[col].dtype),  # default selection
                        # unique ID for each column's selection
                        id=f"datatype-{col}",
                        inline=True,
                        style={
                            "text-align": "center",
                            "margin": "auto",
                            "width": "100%"
                        }
                    ),
                    html.Hr()
                ]
            )
        )

    return selection_options


@app.callback(Output("modal-body-scroll", "is_open", allow_duplicate=True),
              Output("allow-kde", "disabled"),
              Output("allow-kde", "on"),
              Output("modal-body", "children"),
              Input('upload-data', 'contents'),
              Input('use-default-dataset', 'value'),
              State('upload-data', 'filename'),
              prevent_initial_call=True)
def upload_dataset(content, use_default, name):
    df = None
    dataset_name = None
    
    # Check if user wants to use default dataset
    if use_default and 'use_default' in use_default:
        # Load default Asia dataset
        default_path = os.path.join(os.path.dirname(__file__), 'asia_default.csv')
        try:
            df = pd.read_csv(default_path, sep=None, engine='python', na_values='?')
            dataset_name = 'asia_default.csv'
            
            # Apply same preprocessing as parse_content
            df = df.dropna(axis=1, thresh=int(0.3*len(df)))
            df = df.dropna(axis=0)
            
            index_constant = np.where(df.nunique() == 1)[0]
            constant_columns = [df.columns[i] for i in index_constant]
            df = df.drop(columns=constant_columns, axis=1)
            
            cat_data = df.select_dtypes('object').astype('category')
            for c in cat_data:
                df = df.assign(**{c: cat_data[c]})
            
            float_data = df.select_dtypes('number').astype('float64')
            for c in float_data:
                df = df.assign(**{c: float_data[c]})
            
            df.reset_index(drop=True, inplace=True)
            
            to_remove_features = linear_dependent_features(df)
            df = df.drop(columns=to_remove_features, axis=1)
        except Exception as e:
            print(f"Error loading default dataset: {e}")
            return no_update, no_update, no_update, no_update
    
    # If not using default and there's uploaded content, parse it
    elif content is not None:
        df = parse_content(content, name)
        dataset_name = name

    # If we have a valid dataframe, create session and show modal
    if df is not None and dataset_name is not None:
        create_session(df, dataset_name)
        return True, (len(df.select_dtypes('number').columns) == 0), (False if len(df.select_dtypes('number').columns) == 0 else no_update), populate_modal(df)

    return no_update, no_update, no_update, no_update


# Callback to handle and display the selections made
@app.callback(
    Output("modal-body-scroll", "is_open", allow_duplicate=True),
    Output('btn-learn-score-and-search', 'n_clicks'),
    State("modal-body", "children"),
    Input("allow-kde", "on"),
    Input("save-selections", "n_clicks"),
    prevent_initial_call=True
)
def apply_datatypes(selections, allow_kde, *_):
    # Recursive function to find matches
    def find_datatype_ids(obj):
        datatype_pattern = re.compile(r'^datatype-.*$')
        matches = []
        if isinstance(obj, dict):
            for key, value in obj.items():
                if isinstance(value, dict) and 'id' in value and datatype_pattern.match(value['id']):
                    matches.append({value['id']: value['value']})
                else:
                    matches.extend(find_datatype_ids(value))
        elif isinstance(obj, list):
            for item in obj:
                matches.extend(find_datatype_ids(item))
        return matches

    triggered_id = ctx.triggered_id

    if 'save-selections' == triggered_id:

        session_dict = session_objects[session.get('session_id')]
        df = session_dict['temp_df']
        filename = session_dict['temp_filename']

        # Find all "datatype-{col}" items
        datatype_items = find_datatype_ids(selections)
        datatype_items = {k: v for d in datatype_items for k, v in d.items()}
        for col in df.columns:
            final_dtype = datatype_items[f'datatype-{col}']
            if str(df[col].dtype) == 'float64' and final_dtype == 'category':
                df[col] = df[col].astype('str').astype(final_dtype)
            df[col] = df[col].astype(final_dtype)

        last_score = session_dict.get('selected_score', 'BIC')
        last_i_test = session_dict.get('selected_i_test', 'Mutual Information')
        last_layout = session_dict.get('layout', 'Circular')
        session_objects[session.get('session_id')] = dict()
        session_dict = session_objects[session.get('session_id')]
        session_dict['user_df'] = df
        session_dict['filename'] = filename
        session_dict['selected_score'] = last_score
        session_dict['selected_i_test'] = last_i_test
        session_dict['layout'] = last_layout
        session_dict['allow_kde'] = allow_kde
        session_dict['learning_alg'] = 'score_and_search'

        return False, 1

    return no_update, no_update


@app.callback(Output("modal-download-dialogue", "is_open"),

              Input('bt-export', 'n_clicks'),
              prevent_initial_call=True)
def show_export_dialogue(_):

    triggered_id = ctx.triggered_id

    if 'bt-export' == triggered_id:
        return True

    return no_update


@app.callback(
    Output("cytoscape", "generateImage"),

    Input('bt-export-jpg', 'n_clicks'),
    Input('bt-export-png', 'n_clicks'),
    Input('bt-export-svg', 'n_clicks'),
    prevent_initial_call=True
)
def get_image(*_):
    session_dict = session_objects[session.get('session_id')]
    # File type to output of 'svg, 'png', 'jpg'
    triggered_id = ctx.triggered_id

    action = "download"
    ftype = triggered_id.split("-")[-1]

    return {
        'type': ftype,
        'action': action,
        'options': {'full': True, 'scale': 4},
        'filename': session_dict['filename'] + "_" + str(session_dict['bn'].type()).strip('Type'),
    }


def adj_list_to_matrix(adj, n_nodes):

    # Initialize a matrix
    matrix = [[0 for _ in range(n_nodes)]
              for _ in range(n_nodes)]

    for (i, j) in adj:
        matrix[i][j] = 1

    return matrix


@app.callback(Output("download-item", "data"),
              Input('bt-export-csv-m', 'n_clicks'),
              Input('bt-export-csv-l', 'n_clicks'),
              Input('bt-export-csv-types', 'n_clicks'),
              prevent_initial_call=True)
def export_network(*_):
    session_dict = session_objects[session.get('session_id')]
    triggered_id = ctx.triggered_id

    base_name = session_dict['filename'] + "_" + \
        str(session_dict['bn'].type()).strip('Type')

    if 'bt-export-csv-l' == triggered_id:
        if session_dict['learning_alg'] == 'score_and_search':
            arcs = session_dict['bn'].arcs()
        else:
            arcs = session_dict['pdag'].arcs()
            for source, dest in session_dict['pdag'].edges():
                arcs.append((source, dest))
                arcs.append((dest, source))

        exp_df = pd.DataFrame(arcs, columns=[
            'source', 'destination'])

        data = dcc.send_data_frame(
            exp_df.to_csv, base_name + '_adj_list' + ".csv", index=False)
    elif 'bt-export-csv-m' == triggered_id:
        nodes = session_dict['bn'].nodes()
        nodes.sort()
        enc_nodes = {y: x for (x, y) in enumerate(nodes)}
        if session_dict['learning_alg'] == 'score_and_search':
            enc_arcs = [(enc_nodes[u], enc_nodes[v])
                        for (u, v) in session_dict['bn'].arcs()]
        else:
            enc_arcs = [(enc_nodes[u], enc_nodes[v])
                        for (u, v) in session_dict['pdag'].arcs()]
            for source, dest in session_dict['pdag'].edges():
                enc_arcs.append((enc_nodes[source], enc_nodes[dest]))
                enc_arcs.append((enc_nodes[dest], enc_nodes[source]))
        adjMatrix = adj_list_to_matrix(enc_arcs, len(nodes))
        exp_df = pd.DataFrame(adjMatrix, columns=enc_nodes.keys())
        exp_df = pd.concat(
            [pd.DataFrame(np.array(nodes).reshape(-1, 1), columns=['id']), exp_df], axis=1)
        data = dcc.send_data_frame(
            exp_df.to_csv, base_name + '_adj_matrix' + ".csv", index=False)

    elif 'bt-export-csv-types' == triggered_id:
        factor_types = {'factor_name': [], 'factor_type': []}
        for node in session_dict['bn'].nodes():
            factor_types['factor_name'].append(node)
            factor_types['factor_type'].append(
                str(session_dict['bn'].node_type(node)))

        exp_df = pd.DataFrame().from_dict(factor_types)
        data = dcc.send_data_frame(
            exp_df.to_csv, base_name + '_factor_types' + ".csv", index=False)

    return data


@app.callback(Output("select-show-score", "hidden"),
              Output("loglik-graph", "hidden"),
              Output("select-pc-hparams", "hidden"),
              Output("pc-phase-menu", "hidden"),
              Output("pc-phase-text", "children", allow_duplicate=True),
              Output("toggle-positive-operations", "options"),
              Output("toggle-positive-operations", "value"),
              Output("btn-call-library", "children"),
              Output("infinite-sort-filter-grid-2",
                     "getRowsRequest", allow_duplicate=True),
              Output("infinite-sort-filter-grid-2",
                     "columnDefs", allow_duplicate=True),
              Output('score-dropdown', 'value', allow_duplicate=True),
              Output('bt-reset', 'n_clicks', allow_duplicate=True),
              Output('add-whitelist-button', 'n_clicks', allow_duplicate=True),
              Output('total-score', 'children', allow_duplicate=True),
              Output("bt-sepset-increase", "disabled", allow_duplicate=True),
              Output("div-sepset-increase", "hidden", allow_duplicate=True),
              Output('itest-dropdown', 'value', allow_duplicate=True),
              Output('input-alpha', 'value', allow_duplicate=True),
              Output('rcot-hparams', 'hidden', allow_duplicate=True),
              Output('n-uncond', 'value', allow_duplicate=True),
              Output('n-cond', 'value', allow_duplicate=True),
              Output('knncmi-hparams', 'hidden', allow_duplicate=True),
              Output('k-neigh', 'value', allow_duplicate=True),
              Output('k-perm', 'value', allow_duplicate=True),
              Output('samples', 'value', allow_duplicate=True),
              Output('hc-hparams', 'hidden', allow_duplicate=True),
              Output('max-parents', 'value'),
              Output("next-pc-button", "disabled", allow_duplicate=True),
              Output("prev-pc-button", "disabled", allow_duplicate=True),
              Output("div-ambiguous-threshold",
                     "hidden", allow_duplicate=True),
              Output("div-orient", "hidden", allow_duplicate=True),
              Output("div-pc-phase-change", "hidden", allow_duplicate=True),
              Output("alg-phases-title", "children", allow_duplicate=True),
              Output('whitelisted-arcs-grid', 'rowData', allow_duplicate=True),
              Output('blacklisted-arcs-grid', 'rowData', allow_duplicate=True),

              Input('btn-learn-score-and-search', 'n_clicks'),
              Input('btn-learn-constraint-based', 'n_clicks'),
              Input('btn-hybrid-learning', 'n_clicks'),
              prevent_initial_call=True)
def change_learning_algorithm(n_clicks_score_and_search, n_clicks_constraint_based, _):
    triggered_id = ctx.triggered_id
    session_dict = session_objects[session.get('session_id')]
    session_dict['positive'] = True

    selected_score = total_score = next_forbidden = prev_forbidden = fill_arc_table = hide_phase_menu = phases_title = no_update
    render_layout = 0

    getRowsRequest = {'startRow': 0, 'endRow': 100,
                      'sortModel': [], 'filterModel': {}}
    if 'btn-learn-score-and-search' == triggered_id:
        session_dict['learning_alg'] = 'score_and_search'

        if n_clicks_score_and_search != -1:
            session_dict['hybrid_learning'] = False
            fill_arc_table = 0
            hide_phase_menu = True
        else:
            phases_title = 'Current hybrid phase: Wrapper'

        columnDefs = [
            {"field": "Operation"},
            {"field": "Source"},
            {"field": "Target"},
            {"field": "Motivation", "filter": "agNumberColumnFilter",
             'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
        ]
        session_dict['table_colnames'] = [
            'Operation', 'Source', 'Target', 'Motivation']
        setup_alg()
        selected_score = session_dict['selected_score']
        score = session_dict['score']
        total_score = "{:.3f}".format(score.score(session_dict['bn']))

        return False, False, True, hide_phase_menu, no_update, [{'label': 'Only positive operations', 'value': 'option1', 'disabled': False}], ['option1'], 'Apply phase', getRowsRequest, columnDefs, selected_score, render_layout, fill_arc_table, total_score, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update, False, 10, next_forbidden, prev_forbidden, no_update, no_update, no_update, phases_title, session_dict['arc_whitelist'], session_dict['arc_blacklist']
    elif 'btn-learn-constraint-based' == triggered_id:
        session_dict['learning_alg'] = 'constraint_based'
        phases_title = 'Current hybrid phase: Filter'
        apply_all_text = 'Apply phase'
        if n_clicks_constraint_based != -1:
            session_dict['hybrid_learning'] = False
            fill_arc_table = 0
            phases_title = 'Current phase'
            apply_all_text = 'Apply remaining PC phases'
        columnDefs = [
            {"field": "Operation"},
            {"field": "Node1"},
            {"field": "Node2"},
            {"field": "Separating set"},
            {"field": "p-value", "headerName": "p-value", "filter": "agNumberColumnFilter",
             'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
        ]
        session_dict['table_colnames'] = ['Operation',
                                          'Node1', 'Node2', 'Separating set', 'p-value']
        setup_alg()
        selected_i_test = session_dict['selected_i_test']
        next_forbidden = False
        prev_forbidden = True
        default_knn = max(len(session_dict['user_df'])//100, 10)
        return True, True, False, False, ["Adjacency search ", fr'$\ \ell = {session_dict['sepset_size']}$'], [{'label': 'Only significant operations', 'value': 'option1', 'disabled': False}], ['option1'], apply_all_text, getRowsRequest, columnDefs, selected_score, render_layout, fill_arc_table, total_score, False, False, selected_i_test, 0.05, (selected_i_test != 'RCoT (Cont)'), 5, 100, (selected_i_test != 'MixedKnnCMI'), default_knn, default_knn, 10, True, no_update, next_forbidden, prev_forbidden, True, True, False, phases_title, session_dict['arc_whitelist'], session_dict['arc_blacklist']
    else:
        session_dict['learning_alg'] = 'constraint_based'
        session_dict['hybrid_learning'] = True
        columnDefs = [
            {"field": "Operation"},
            {"field": "Node1"},
            {"field": "Node2"},
            {"field": "Separating set"},
            {"field": "p-value", "headerName": "p-value", "filter": "agNumberColumnFilter",
             'valueFormatter': {"function": """d3.format(",.3f")(params.value)"""}},
        ]
        session_dict['table_colnames'] = ['Operation',
                                          'Node1', 'Node2', 'Separating set', 'p-value']
        session_dict['arc_blacklist'] = []
        session_dict['arc_whitelist'] = []
        setup_alg()
        selected_i_test = session_dict['selected_i_test']
        next_forbidden = False
        prev_forbidden = True
        default_knn = max(len(session_dict['user_df'])//100, 10)
        return True, True, False, False, ["Adjacency search ", fr'$\ \ell = {session_dict['sepset_size']}$'], [{'label': 'Only significant operations', 'value': 'option1', 'disabled': False}], ['option1'], 'Apply phase', getRowsRequest, columnDefs, selected_score, render_layout, fill_arc_table, total_score, False, False, selected_i_test, 0.05, (selected_i_test != 'RCoT (Cont)'), 5, 100, (selected_i_test != 'MixedKnnCMI'), default_knn, default_knn, 10, True, no_update, next_forbidden, prev_forbidden, True, True, False, 'Current hybrid phase: Filter', session_dict['arc_whitelist'], session_dict['arc_blacklist']


@app.callback(
    Output("page-load", "data"),
    Input("page-load", "data")
)
def detect_refresh(page_load_data):
    # If data is None, it indicates a refresh or first load
    if page_load_data is None and session.get('session_id', False):
        session.pop('session_id')

    return True
