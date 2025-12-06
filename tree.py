import io
import base64
import dash
from dash import dcc, html, Input, Output, State, callback_context as ctx
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import visdcc
import textwrap

# ---------- Helper Functions ----------
def load_builtin(name):
    """Dataset load"""
    if name == "Iris":
        d = datasets.load_iris(as_frame=True)
    elif name == "Wine":
        d = datasets.load_wine(as_frame=True)
    elif name == 'Breast Cancer':
        d = datasets.load_breast_cancer(as_frame=True)
    else:
        raise ValueError("Unknown dataset")
    df = pd.concat([d.frame.iloc[:, :-1], d.frame.iloc[:, -1]], axis=1)
    df.columns = list(d.feature_names) + ["target"]
    return df, "target", [str(x) for x in d.target_names]

def parse_uploaded(contents, filename):
    """CSV"""
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    try:
        df = pd.read_csv(io.StringIO(decoded.decode('utf-8')))
    except Exception as e:
        raise ValueError(f"Could not parse uploaded file: {e}")
    return df

def _wrap_label(text, max_chars=18):

    if text is None:
        return ""

    text = " ".join(str(text).split())

    return textwrap.fill(text, width=max_chars)

def build_visdcc_tree_from_sklearn(clf, feature_names, class_names):
    n_nodes = clf.tree_.node_count
    children_left = clf.tree_.children_left
    children_right = clf.tree_.children_right
    feature = clf.tree_.feature
    threshold = clf.tree_.threshold
    impurity = clf.tree_.impurity
    samples = clf.tree_.weighted_n_node_samples
    values = clf.tree_.value

    #
    class_colors = [
        '#FF6B6B',  # Red
        '#4ECDC4',  # Teal
        '#45B7D1',  # Blue
        '#FFA07A',  # Light Salmon
        '#98D8C8',  # Mint
        '#F7DC6F',  # Yellow
        '#BB8FCE',  # Purple
        '#85C1E2',  # Sky Blue
        '#F8B739',  # Orange
        '#52B788',  # Green
    ]

    tree_split = feature >= 0
    nodes = []
    
    for i in range(n_nodes):
        node = {
            'id': i, 
            'hidden': False, 
            'show_leaf': True, 
            'fixed': {'y': True}
        }
        
        # 
        title = f"Gini impurity = {impurity[i]:.4f}<br>samples = {int(samples[i])}<br>"
        node['title'] = f"<div style='text-align:center'>{title}</div>"
        
        if tree_split[i]:
            # 内部节点
            feat_name = feature_names[feature[i]] if feature[i] < len(feature_names) else f"f{feature[i]}"
            raw_label = f"{feat_name} > {threshold[i]:.3f}"
            # 
            node['label'] = _wrap_label(raw_label, max_chars=18)
            node['shape'] = 'ellipse'
            node['color'] = 'NavajoWhite'
        else:
            # 
            node['shape'] = 'box'
            class_counts = values[i][0].astype(int)
            if class_counts.sum() == 0:
                node_label = "leaf"
                node['color'] = 'lightgray'
            else:
                majority_idx = int(np.argmax(class_counts))
                node_label = class_names[majority_idx] if majority_idx < len(class_names) else str(majority_idx)
                # 
                node['color'] = class_colors[majority_idx % len(class_colors)]
            node['label'] = _wrap_label(node_label, max_chars=18)
            
        nodes.append(node)

    edges = []
    for i in np.where(tree_split)[0]:
        left = int(children_left[i])
        right = int(children_right[i])
        edges.append({
            'id': f"{i}-{left}", 
            'hidden': False, 
            'from': int(i), 
            'to': left,
            'color': {'color': 'gray', 'inherit': 'from'}, 
            'title': 'Yes', 
            'width': 1
        })
        edges.append({
            'id': f"{i}-{right}", 
            'hidden': False, 
            'from': int(i), 
            'to': right,
            'color': {'color': 'gray', 'inherit': 'from'}, 
            'title': 'No', 
            'width': 1
        })

    return {'nodes': nodes, 'edges': edges}

def get_correct_incorrect_samples(clf, X, y):
    """"""
    y_pred = clf.predict(X)
    correct_idx = np.where(y == y_pred)[0].tolist()
    incorrect_idx = np.where(y != y_pred)[0].tolist()
    return correct_idx, incorrect_idx

# ---------- Dash App ----------
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
server = app.server

vis_options = {
    'height': '750px',
    'layout': {
        'hierarchical': {
            'enabled': True, 
            'sortMethod': 'directed',
            
        'levelSeparation': 100,   
        #'nodeSpacing': 50,       #
        #'treeSpacing': 200,       
        }
    },
    'interaction': {'hover': True},
    'nodes': {
        'shape': 'box', 
        'font': {'size': 12, 'multi': True},   
        'border-radius': '10px',

        'widthConstraint': {
            'maximum': 220   
        },
        
        'margin': 10
    },
    'edges': {
        'arrows': 'None', 
        'smooth': {
            'type': "cubicBezier", 
            'forceDirection': 'vertical'
        }
    },
    'physics': {
        'barnesHut': {
            'avoidOverlap': 0.4
        }
    }
}

# ---------- Layout ----------
app.layout = dbc.Container([
    dbc.Row(dbc.Col(html.H2("Decision Tree Path Visualizer", className="text-center mt-3 mb-4"), width=12)),
    
    dbc.Row([
        # Left Control Panel
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Dataset Settings")),
                dbc.CardBody([
                    html.Label("Select Dataset", className="fw-bold"),
                    dcc.Dropdown(
                        id='dataset-select',
                        options=[
                            {'label': 'Iris (Built-in)', 'value': 'Iris'},
                            {'label': 'Wine (Built-in)', 'value': 'Wine'},
                            {'label': 'Breast Cancer (Built-in)', 'value': 'Breast Cancer'},
                            {'label': 'Upload CSV File', 'value': 'Upload'}
                        ],
                        value='Iris'
                    ),
                    html.Br(),
                    dcc.Upload(
                        id='upload-csv',
                        children=html.Div([
                            'Drag and Drop CSV here or Click to Upload',
                            html.Br(),
                            html.Small('(Only when "Upload CSV File" is selected)', className="text-muted")
                        ]),
                        style={
                            'width': '100%',
                            'height': '60px',
                            'lineHeight': '20px',
                            'borderWidth': '1px',
                            'borderStyle': 'dashed',
                            'borderRadius': '5px',
                            'textAlign': 'center',
                            'padding': '10px'
                        }
                    ),
                    html.Div(id='upload-info', style={'fontSize': 12, 'color': '#555', 'marginTop': '5px'}),
                    html.Br(),
                    html.Label("Target Column (for CSV)", className="fw-bold"),
                    dcc.Dropdown(
                        id='target-column',
                        options=[],
                        placeholder='Select target column'
                    ),
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Model Parameters")),
                dbc.CardBody([
                    dbc.Row([
                        dbc.Col([
                            html.Label("max_depth"),
                            dcc.Input(id='max-depth', type='number', value=3, min=1, className="form-control")
                        ], width=6),
                        dbc.Col([
                            html.Label("criterion"),
                            dcc.Dropdown(
                                id='criterion',
                                options=[
                                    {'label': 'gini', 'value': 'gini'},
                                    {'label': 'entropy', 'value': 'entropy'}
                                ],
                                value='gini'
                            )
                        ], width=6)
                    ]),
                    html.Br(),
                    html.Label("random_state"),
                    dcc.Input(id='random-state', type='number', value=42, className="form-control"),
                    html.Br(),
                    dbc.Button("Train Model", id='train-btn', color='primary', className="w-100")
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Path Visualization")),
                dbc.CardBody([
                    html.Label("Visualization Type", className="fw-bold"),
                    dcc.RadioItems(
                        id='path-type',
                        options=[
                            {'label': ' Custom Sample Indices', 'value': 'custom'},
                            {'label': ' Correctly Classified', 'value': 'correct'},
                            {'label': ' Incorrectly Classified', 'value': 'incorrect'}
                        ],
                        value='custom',
                        labelStyle={'display': 'block', 'marginBottom': '8px'}
                    ),
                    html.Br(),
                    html.Div(id='sample-list-container', children=[
                        html.Label("Available Indices:", className="fw-bold", style={'fontSize': '12px'}),
                        dcc.Dropdown(
                            id='sample-list',
                            options=[],
                            multi=True,
                            placeholder='Select samples to visualize',
                            style={'fontSize': '12px'}
                        ),
                        html.Br(),
                    ]),
                    html.Label("Or Enter Indices (comma separated)", className="fw-bold"),
                    dcc.Input(
                        id='sample-indices',
                        type='text',
                        placeholder='e.g., 0,5,10',
                        className="form-control",
                        disabled=False
                    ),
                    html.Br(),
                    dbc.Button("Highlight Paths", id='highlight-btn', color='success', className="w-100 mb-2"),
                    dbc.Button("Reset Styling", id='reset-btn', color='secondary', className="w-100")
                ])
            ], className="mb-3"),
            
            dbc.Card([
                dbc.CardHeader(html.H5("Model Statistics")),
                dbc.CardBody([
                    html.Pre(id='metrics-output', style={'fontSize': '13px', 'whiteSpace': 'pre-wrap'})
                ])
            ])
            
        ], width=4),
        
        # Right Side Tree Visualization
        dbc.Col([
            dbc.Card([
                dbc.CardHeader(html.H5("Decision Tree Structure (Interactive)")),
                dbc.CardBody([
                    visdcc.Network(
                        id='net',
                        data={'nodes': [], 'edges': []},
                        options=vis_options,
                        selection={'nodes': [], 'edges': []}
                    )
                ])
            ])
        ], width=8)
    ])
], fluid=True)

# ---------- Callbacks ----------

@app.callback(
    Output('target-column', 'options'),
    Output('upload-info', 'children'),
    Input('upload-csv', 'contents'),
    State('upload-csv', 'filename')
)
def update_target_options(contents, filename):
    """Update target column options for CSV file"""
    if contents is None:
        return [], ""
    try:
        df = parse_uploaded(contents, filename)
        options = [{'label': col, 'value': col} for col in df.columns]
        info = f"Uploaded: {filename} ({len(df)} rows, {len(df.columns)} columns)"
        return options, info
    except Exception as e:
        return [], f"Error: {str(e)}"

@app.callback(
    Output('sample-indices', 'disabled'),
    Output('sample-list', 'options'),
    Output('sample-list', 'value'),
    Input('path-type', 'value'),
    State('path-type', 'value')
)
def update_sample_controls(path_type, current_path_type):
    """Update sample controls based on visualization type"""
    if 'CURRENT_MODEL' not in globals():
        return False, [], []
    
    model = CURRENT_MODEL
    
    if path_type == 'correct':
        indices = model.get('correct_idx', [])
        options = [{'label': f"Sample {i}", 'value': i} for i in indices]
        return True, options, []
    elif path_type == 'incorrect':
        indices = model.get('incorrect_idx', [])
        options = [{'label': f"Sample {i}", 'value': i} for i in indices]
        return True, options, []
    else:  # custom
        return False, [], []

@app.callback(
    Output('net', 'data'),
    Output('metrics-output', 'children'),
    Input('train-btn', 'n_clicks'),
    Input('highlight-btn', 'n_clicks'),
    Input('reset-btn', 'n_clicks'),
    State('dataset-select', 'value'),
    State('upload-csv', 'contents'),
    State('upload-csv', 'filename'),
    State('target-column', 'value'),
    State('max-depth', 'value'),
    State('criterion', 'value'),
    State('random-state', 'value'),
    State('path-type', 'value'),
    State('sample-indices', 'value'),
    State('sample-list', 'value'),
    State('net', 'data'),
    prevent_initial_call=True
)
def handle_all_actions(train_n, highlight_n, reset_n,
                       dataset_value, upload_contents, upload_filename, target_col,
                       max_depth, criterion, random_state,
                       path_type, sample_indices, sample_list, current_net_data):
    """Handle all actions: train, highlight, reset"""
    
    triggered = ctx.triggered[0]['prop_id'].split('.')[0] if ctx.triggered else None
    
    # ========== Train Model ==========
    if triggered == 'train-btn':
        try:
            # Load data
            if dataset_value in ('Iris', 'Wine', 'Breast Cancer'):
                df, target_name, class_names = load_builtin(dataset_value)
                X = df.drop(columns=[target_name])
                y = df[target_name]
            else:
                if upload_contents is None:
                    return dash.no_update, "Please upload a CSV file and select target column"
                df = parse_uploaded(upload_contents, upload_filename)
                if target_col is None:
                    return dash.no_update, "Please select the target column"
                df = df.dropna(subset=[target_col])
                y = df[target_col]
                X = df.drop(columns=[target_col])
                class_names = [str(x) for x in sorted(pd.unique(y))]

            # Numeric conversion
            X_numeric = X.select_dtypes(include=[np.number]).copy()
            non_numeric = [c for c in X.columns if c not in X_numeric.columns]
            for c in non_numeric:
                try:
                    X_numeric[c] = pd.factorize(X[c])[0]
                except Exception:
                    pass
            X = X_numeric.select_dtypes(include=[np.number])
            
            if X.shape[1] == 0:
                return dash.no_update, "No numeric features available for training"

            # Train-test split
            rs = int(random_state) if random_state is not None else None
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.25, random_state=rs,
                stratify=y if len(pd.unique(y)) > 1 else None
            )

            # Train model
            max_depth_arg = None if (max_depth is None or max_depth == '') else int(max_depth)
            clf = DecisionTreeClassifier(
                max_depth=max_depth_arg,
                criterion=criterion,
                random_state=rs
            )
            clf.fit(X_train, y_train)

            # Calculate accuracy
            y_train_pred = clf.predict(X_train)
            y_test_pred = clf.predict(X_test)
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)

            # Get correct and incorrect samples
            X_all = pd.concat([X_train, X_test]).reset_index(drop=True)
            y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
            correct_idx, incorrect_idx = get_correct_incorrect_samples(clf, X_all, y_all)

            # Build visualization data
            feature_names = list(X.columns)
            data = build_visdcc_tree_from_sklearn(clf, feature_names, class_names)

            # Output statistics
            metrics_text = (
                f"Dataset: {dataset_value}\n"
                f"Samples: {len(X)}\n"
                f"Features: {X.shape[1]}\n"
                f"Classes: {len(class_names)}\n"
                f"max_depth: {max_depth_arg}\n"
                f"criterion: {criterion}\n\n"
                f"Train Accuracy: {train_acc:.4f}\n"
                f"Test Accuracy: {test_acc:.4f}\n\n"
                f"Correctly Classified: {len(correct_idx)}\n"
                f"Incorrectly Classified: {len(incorrect_idx)}\n\n"
                f"Correct indices (first 20):\n{correct_idx[:20]}\n\n"
                f"Incorrect indices (all):\n{incorrect_idx}"
            )

            # Store global variables for later use
            global CURRENT_MODEL
            CURRENT_MODEL = {
                'clf': clf,
                'X': X_all,
                'y': y_all,
                'feature_names': feature_names,
                'class_names': class_names,
                'correct_idx': correct_idx,
                'incorrect_idx': incorrect_idx,
                'metrics': metrics_text
            }

            return data, metrics_text

        except Exception as e:
            return dash.no_update, f"Training failed: {str(e)}"

    # ========== Highlight Paths ==========
    elif triggered == 'highlight-btn':
        if 'clf' not in globals().get('CURRENT_MODEL', {}):
            return dash.no_update, "Please train the model first"

        clf = CURRENT_MODEL['clf']
        X = CURRENT_MODEL['X']
        
        # Determine which samples to highlight
        if path_type == 'correct':
            if sample_list and len(sample_list) > 0:
                valid_idxs = sample_list
                desc = f"Correctly Classified (Selected {len(sample_list)})"
            else:
                valid_idxs = CURRENT_MODEL['correct_idx']
                desc = "Correctly Classified (All)"
        elif path_type == 'incorrect':
            if sample_list and len(sample_list) > 0:
                valid_idxs = sample_list
                desc = f"Incorrectly Classified (Selected {len(sample_list)})"
            else:
                valid_idxs = CURRENT_MODEL['incorrect_idx']
                desc = "Incorrectly Classified (All)"
        else:  # custom
            if not sample_indices:
                return dash.no_update, "Please enter sample indices"
            try:
                idxs = [int(x.strip()) for x in sample_indices.split(',') if x.strip() != '']
                valid_idxs = [i for i in idxs if 0 <= i < len(X)]
                desc = "Custom Samples"
            except Exception:
                return dash.no_update, "Invalid sample indices format. Use comma-separated integers"

        if not valid_idxs:
            return dash.no_update, "No valid sample indices"

        # Reset edge styling
        data = current_net_data or {'nodes': [], 'edges': []}
        for e in data['edges']:
            e['width'] = 1
            e['color'] = {'color': 'gray', 'inherit': 'from'}
            e.pop('shadow', None)

        # Count edge usage
        edge_counts = {}
        for sample_idx in valid_idxs:
            xi = X.iloc[[sample_idx]].values
            dp = clf.decision_path(xi)
            node_idx = dp.indices[dp.indptr[0]:dp.indptr[1]]
            for j in range(len(node_idx) - 1):
                edge_id = f"{node_idx[j]}-{node_idx[j+1]}"
                edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1

        # Set different colors based on path type
        if path_type == 'correct':
            highlight_color = 'rgba(0,170,162,1.0)'  # Teal - Correct
            shadow_color = 'rgba(0,170,162,0.9)'
        elif path_type == 'incorrect':
            highlight_color = 'rgba(200,0,0,1.0)'  # Red - Incorrect
            shadow_color = 'rgba(200,0,0,0.9)'
        else:
            highlight_color = 'rgba(0,100,255,1.0)'  # Blue - Custom
            shadow_color = 'rgba(0,100,255,0.9)'

        # Apply highlighting with adjusted width calculation
        max_count = max(edge_counts.values()) if edge_counts else 1
        for eid, cnt in edge_counts.items():
            for e in data['edges']:
                if e['id'] == eid:
                    # Adjust width calculation: use sqrt to reduce growth rate
                    # Maximum width is 10, minimum is 2
                    normalized_width = 2 + (8 * np.sqrt(cnt / max_count))
                    e['width'] = normalized_width
                    e['color'] = {'color': highlight_color, 'inherit': 'from'}
                    e['shadow'] = {
                        'enabled': True,
                        'color': shadow_color,
                        'size': 0.5,
                        'y': 5
                    }
                    break

        info = f"\n\n=== Path Highlighted ===\n{desc}\nNumber of samples: {len(valid_idxs)}\nSample indices: {valid_idxs[:50]}"
        if len(valid_idxs) > 50:
            info += f"\n... and {len(valid_idxs) - 50} more"
        
        return data, CURRENT_MODEL.get('metrics', '') + info

    # ========== Reset Styling ==========
    elif triggered == 'reset-btn':
        data = current_net_data or {'nodes': [], 'edges': []}
        for e in data['edges']:
            e['width'] = 1
            e['color'] = {'color': 'gray', 'inherit': 'from'}
            e.pop('shadow', None)
        
        return data, CURRENT_MODEL.get('metrics', '') if 'CURRENT_MODEL' in globals() else "Please train the model first"

    return dash.no_update, dash.no_update


if __name__ == '__main__':
    app.run_server(debug=True, port=8050)
