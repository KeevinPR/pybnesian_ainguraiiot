from dash import html, dcc, jupyter_dash
import dash_bootstrap_components as dbc
import dash_ag_grid as dag
import dash_daq as daq
import dash_cytoscape as cyto

from flask_caching import Cache  # Import Cache for caching

import dash_ag_grid as dag
import dash_cytoscape as cyto


from callbacks import *


dag.AgGrid(
    rowModelType="infinite",
)

notification_factor = dbc.Toast(
    [
        html.P(id="node-factor-text", children="",
               style={'textAlign': 'center', 'font-size': '24px'})
    ],
    id="node-factor-notification",
    header="Factor Type",
    duration=4000,
    is_open=False,
    dismissable=True,
    style={
        "position": "absolute",
        "bottom": "10px",
        "right": "10px",
        "width": "20%",
        "zIndex": 1000,  # Ensure it's above other elements
    }
)

layout_menu = html.Div([dbc.Button('Re-render layout', id='bt-reset', style={"margin": 20}, className="me-1", color="secondary", outline=True),
                        dcc.Dropdown(list(layout_dict.keys()), 'Circular', searchable=False, clearable=False,
                                     id='layout-dropdown', style={"margin": 10, "width": "250px",  "padding": "0px"}),
                        html.Div([
                            dbc.Button('Export Structure', id='bt-export',
                                       style={"margin": 20}, className="me-1", color="primary", outline=True)
                        ], style={'textAlign': 'center'}),

                        html.Div([
                            dcc.Upload(
                                id='upload-data',
                                children=dbc.Button('Upload CSV', id='upload-button', style={
                                    "margin": 20}, className="me-1", color="primary", outline=True),
                                multiple=False  # Only allow one file
                            )
                        ], style={'textAlign': 'center'})], style={'display': 'flex', 'flexDirection': 'row', 'justify-content': 'space-between'})

cyto_board = html.Div(
    [
        html.Div(
            [
                cyto.Cytoscape(
                    id="cytoscape",
                    elements=[],
                    style={"width": "70%", "height": "900px"},
                    # "preset" to use the pos coords
                    layout={"name": "preset"},
                    stylesheet=default_stylesheet,
                    zoomingEnabled=True,
                ),
                # Toast Notification positioned absolutely in the bottom right
                notification_factor], style={"position": "relative", "margin": 10, 'flex': 1, "border": "2px white solid", "border-radius": "5px"})
    ], style={"margin": 5, 'flex': 1, "border": "2px black solid", "border-radius": "15px"})

loglik_graph = html.Div(id='loglik-graph', children=[dcc.Graph(id='logl-graph', animate=True, mathjax=True, responsive=True, figure={'data': [plotly.graph_objs.Scatter(
    x=[0],
    y=[0],
    name='Scatter',
    mode='lines+markers',


)], 'layout': plotly.graph_objs.Layout(xaxis=dict(range=[0, 0],
                                                  title="Steps"),
                                       yaxis=dict(range=[0, 0],
                                                  title=r'$\mathcal{L}\left(\theta,\mathcal{G}\mid\mathcal{D}\right)\newline$'),
                                       title='Log-likelihood of the structure'
                                       )})])

select_learning_alg = html.Div([dbc.Button("Score and Search (Hill Climbing)",
                                           id="btn-learn-score-and-search", style={"margin": 10, 'align-self': 'center'}, className="me-1", color="secondary", outline=True),
                                dbc.Button("Constraint-based (MPC-stable)",
                                           id="btn-learn-constraint-based", style={"margin": 10, 'align-self': 'center'}, className="me-1", color="secondary", outline=True),
                                dbc.Button("Hybrid Learning",
                                           id="btn-hybrid-learning", style={"margin": 10, 'align-self': 'center'}, className="me-1", color="secondary", outline=True)], style={
    'display': 'flex', 'justify-content': 'center',  # Center the outer div
    'width': '100%', 'margin': 10
})

cvlikelihood_hparams = html.Div(
    id='cvlikelihood-hparams',
    hidden=True,
    children=[
        html.Div([
            html.Label("k-folds", style={'margin-right': 10}),
            dcc.Input(
                id="k-folds",
                type="number",
                step=1,
                min=2,
                max=10,
                value=5,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),  # Spacing between inputs
    ], style={'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'row'}
)

hc_hparams = html.Div(
    id='hc-hparams',
    hidden=False,
    children=[
        html.Div([
            html.Label("max-parents", style={'margin-right': 10}),
            dcc.Input(
                id="max-parents",
                type="number",
                step=1,
                min=1,
                value=10,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),  # Spacing between inputs
    ], style={'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'row'}
)


rcot_hparams = html.Div(
    id='rcot-hparams',
    hidden=True,
    children=[
        html.Div([
            html.Label("n-uncond", style={'margin-right': 10}),
            dcc.Input(
                id="n-uncond",
                type="number",
                step=1,
                min=1,
                value=5,
                max=50,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),
        html.Div([
            html.Label("n-cond", style={'margin-right': 10}),
            dcc.Input(
                id="n-cond",
                type="number",
                step=1,
                min=10,
                max=300,
                value=100,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),


    ], style={'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'row'}
)

knncmi_hparams = html.Div(
    id='knncmi-hparams',
    hidden=True,
    children=[
        html.Div([
            html.Label("k-neigh", style={'margin-right': 10}),
            dcc.Input(
                id="k-neigh",
                type="number",
                step=1,
                min=1,
                value=10,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),
        html.Div([
            html.Label("k-perm", style={'margin-right': 10}),
            dcc.Input(
                id="k-perm",
                type="number",
                step=1,
                min=1,
                value=10,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),
        html.Div([
            html.Label("samples", style={'margin-right': 10}),
            dcc.Input(
                id="samples",
                type="number",
                step=1,
                min=3,
                value=10,
                inputMode='numeric',
                debounce=2
            )
        ], style={'margin': 10}),


    ], style={'justify-content': 'space-between', 'display': 'flex', 'flexDirection': 'row'}
)
select_show_score = html.Div(id='select-show-score', hidden=False,

                             children=[hc_hparams,
                                       dcc.Dropdown(['BIC',
                                                     'CVLikelihood',
                                                     'BDe/BGe (Homog)',
                                                     ], 'BIC', searchable=False, clearable=False,
                                                    id='score-dropdown', style={"margin": 10, "width": "180px", "padding": "0px"}),
                                       cvlikelihood_hparams,

                                       html.P('Score:', style={"margin": 10, 'align-self': 'center'}), html.Pre(id='total-score', style={
                                           'margin': '0px',
                                           'padding': '5px',
                                           'padding-top': '8px',
                                           'padding-bottom': '8px',
                                           'border-radius': '5px',
                                           'border': '2px solid #00A2E1',
                                           'background-color': '#f0f8ff',
                                           'text-align': 'center',
                                           'font-size': '14px'
                                       })],
                             style={'display': 'flex', 'justify-content': 'flex-end',
                                    'align-items': 'center', 'flexDirection': 'row'}
                             )

select_pc_hparams = html.Div(id='select-pc-hparams', hidden=True,

                             children=[
                                 html.Div([
                                     html.Label(
                                         "Independence test", style={'margin-right': 10}),
                                     dcc.Dropdown([
                                         'Mutual Information',
                                         'LinearCorr (Cont)',
                                         'RCoT (Cont)',
                                         'χ2 (Discr)',
                                         'MixedKnnCMI'
                                     ], 'Mutual Information', searchable=False, clearable=False,
                                         id='itest-dropdown', style={"margin": 10, "width": "200px", "padding": "0px"}),

                                     rcot_hparams,
                                     knncmi_hparams

                                 ], style={'margin': 10, 'display': 'flex', 'justify-content': 'flex-end',
                                           'align-items': 'center', 'flexDirection': 'row'}),
                                 html.Div([
                                     html.Label(
                                         "alpha", style={'margin-right': 10}),
                                     dcc.Input(
                                         id="input-alpha",
                                         type="number",
                                         min=0.0001,
                                         step=0.0001,
                                         max=0.9999,
                                         value=0.05,
                                         inputMode='numeric',
                                         debounce=2
                                     )
                                 ], style={'margin': 10, 'display': 'flex', 'justify-content': 'flex-end',
                                           'align-items': 'center', 'flexDirection': 'row'}),
                                 html.Div(id='div-ambiguous-threshold', children=[
                                     html.Label(
                                         "ambiguous-threshold", style={'margin-right': 10}),
                                     dcc.Input(
                                         id="input-ambiguous-threshold",
                                         type="number",
                                         min=0.0001,
                                         step=0.0001,
                                         max=0.9999,
                                         value=0.5,
                                         inputMode='numeric',
                                         debounce=2
                                     )
                                 ],  hidden=True, style={'margin': 10, 'display': 'flex', 'justify-content': 'flex-end',
                                                         'align-items': 'center', 'flexDirection': 'row'})],
                             style={'justify-content': 'space-between',
                                    'display': 'flex', 'flexDirection': 'row'}
                             )


operation_header = html.Div([html.P('Please select one:', style={"margin": 10, 'align-self': 'center'}),
                             dcc.Checklist([{'label': 'Only positive operations', 'value': 'option1', 'disabled': False}], ['option1'], id="toggle-positive-operations",
                                           style={"margin": 10, 'align-self': 'center'}),
                             select_show_score,
                             select_pc_hparams


                             ],
                            style={'display': 'flex', 'flexDirection': 'row', "margin": 20, 'justify-content': 'space-between'})

operation_selection = html.Div(dcc.Loading(
    delay_show=1000,
    id="loading-deltas",
    type="default",
    children=dag.AgGrid(
        id="infinite-sort-filter-grid-2",
        columnSize="sizeToFit",
        columnDefs=column_definitions_dict['score_and_search'],
        defaultColDef={"sortable": True,
                       "filter": True, "floatingFilter": True,
                       "resizable": True,
                       "flex": 1, },
        rowModelType="infinite",
        dashGridOptions={
            # The number of rows rendered outside the viewable area the grid renders.
            "rowBuffer": 0,
            # How many blocks to keep in the store.
            "maxBlocksInCache": 1,
            "rowSelection": "single",
            "suppressCellFocus": True,
            "enableCellTextSelection": True,
            "ensureDomOrder": True,
        },
        className="ag-theme-balham",
    )
), style={"margin": 15, 'flex': 1})


speed_slider = html.Div(
    dcc.Slider(
        600, 1500,
        id='speed-slider',
        step=100,
        marks=None,
        value=1000,
        tooltip={"placement": "bottom",
                 "always_visible": False}
    ),
    # Set width for the slider container
    style={'width': '80%', 'margin': 'auto'}
)

pc_phase_menu = html.Div(
    id="pc-phase-menu",
    hidden=True,
    children=[
        # Centered title text
        html.Div(
            "Current PC phase",
            id='alg-phases-title',
            style={'fontSize': '24px',
                   'fontWeight': 'bold', 'textAlign': 'center'}
        ),

        # Threshold text and a hiddable button next to it
        html.Div(
            style={
                'display': 'flex',
                'alignItems': 'center',  # Vertically centers the items
                # Centers the items horizontally in the container
                'justifyContent': 'space-between',
                'marginTop': '10px',

            },
            children=[
                dcc.Markdown(["Adjacency search ", r'$\ \ell = 0$'], mathjax=True, style={
                             'marginTop': '15px', 'fontSize': '20px'}, id="pc-phase-text"),
                html.Div(
                    dbc.Button("Increase", id="bt-sepset-increase", className="me-1", color="secondary", outline=True, disabled=False, style={'margin': 10}), id="div-sepset-increase"),
                html.Div(
                    dbc.Button("Fully orient", id="bt-orient", className="me-1", color="secondary", outline=True, disabled=True, style={'margin': 10}), id="div-orient", hidden=True)
            ]
        ),

        # Side-by-side buttons below
        html.Div(id='div-pc-phase-change',
                 style={'display': 'flex',
                        'justifyContent': 'space-between', 'marginTop': '20px'},
                 children=[
                     dbc.Button("Previous phase", id="prev-pc-button", className="me-1",
                                color="secondary", outline=True, disabled=True),
                     dbc.Button("Next phase", id="next-pc-button",
                                className="me-1", color="secondary", outline=True)
                 ]
                 )
    ],
    style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'justifyContent': 'center',
    }
)
apply_operation_menu = html.Div(

    children=[pc_phase_menu, html.Div([html.Div([
        dbc.Button("Apply operation",
                   id="btn-row-selection-apply", style={"margin": 10, 'align-self': 'center'}, className="me-1", color="secondary", outline=True),
        html.P('autorun:',
               style={"margin": 10, 'align-self': 'center'}),
        daq.PowerButton(
            id='start-stop-button',
            on=False,
            color='#00A2E1',
            size=50,
            style={"margin": 10, 'align-self': 'center'}
        ), dbc.Button("Apply phase",
                      id="btn-call-library", style={"margin": 10, 'align-self': 'center'}, className="me-1", color="secondary", outline=True),
    ],
        style={'display': 'flex', 'flexDirection': 'row', 'text-align': 'left'}),
        speed_slider,

        html.P('Delay (ms)', style={"margin": 0, 'text-align': 'center'}),

    ], style={
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'center',
        'width': '100%',
        'max-width': '50%'
    })], style={
        'display': 'flex', 'justify-content': 'center',  # Center the outer div
        'width': '100%', 'margin': 20
    })


arc_constraint_lists = html.Div([

    dbc.Col([
        html.H4("Arc constraints"),
        dag.AgGrid(
            id='select-arcs-grid',
            columnDefs=[
                {"headerName": "Source", "field": "Source",
                 "checkboxSelection": True},
                {"headerName": "Target", "field": "Target"},
            ],
            rowData=[],
            columnSize="sizeToFit",
            style={'height': '200px', 'width': '400px'},
            dashGridOptions={
                "rowSelection": "multiple",
                "suppressCellFocus": True,
                "ensureDomOrder": True,
            },
            className="ag-theme-balham",
            defaultColDef={"sortable": True,
                           "filter": True,
                           "floatingFilter": True},

        ),
        dbc.Button("Add to whitelist", id="add-whitelist-button",
                   n_clicks=0, style={'marginTop': '10px'}, className="me-1", color="secondary", outline=True),
        dbc.Button("Add to blacklist", id="add-blacklist-button",
                   n_clicks=0, style={'marginTop': '10px'}, className="me-1", color="secondary", outline=True),

    ], width="33%"),

    dbc.Col([
        html.H4("Whitelist"),
        dag.AgGrid(
            id='whitelisted-arcs-grid',
            columnDefs=[
                {"headerName": "Source", "field": "Source",
                 "checkboxSelection": True},
                {"headerName": "Target", "field": "Target"},
            ],
            rowData=[],
            columnSize="sizeToFit",
            style={'height': '200px', 'width': '400px'},
            dashGridOptions={
                "rowSelection": "multiple",
                "suppressCellFocus": True,
                "ensureDomOrder": True,
            },
            className="ag-theme-balham",
            defaultColDef={"sortable": True,
                           "filter": True,
                           "floatingFilter": True},

        ),
        dbc.Button("Remove", id="remove-whitelist-button",
                   n_clicks=0, style={'marginTop': '10px'}, className="me-1", color="secondary", outline=True),
    ], width="33%", style={'marginLeft': 20}),
    dbc.Col([
        html.H4("Blacklist"),
        dag.AgGrid(
            id='blacklisted-arcs-grid',
            columnDefs=[
                {"headerName": "Source", "field": "Source",
                 "checkboxSelection": True},
                {"headerName": "Target", "field": "Target"},
            ],
            rowData=[],
            columnSize="sizeToFit",
            style={'height': '200px', 'width': '400px'},
            dashGridOptions={
                "rowSelection": "multiple",
                "suppressCellFocus": True,
                "ensureDomOrder": True,
            },
            className="ag-theme-balham",
            defaultColDef={"sortable": True,
                           "filter": True,
                           "floatingFilter": True},

        ),
        dbc.Button("Remove", id="remove-blacklist-button",
                   n_clicks=0, style={'marginTop': '10px'}, className="me-1", color="secondary", outline=True),
    ], width="33%", style={'marginLeft': 20}),


], style={'display': 'flex', 'flexDirection': 'row', 'justify-content': 'center'})

error_notification = dbc.Toast(
    [html.P("", id="notification-text", className="mb-0")],
    id="notification",
    header="Error",
    icon="danger",
    dismissable=True,
    is_open=False,
    duration=10000,
    style={

        'marginTop': '10px',
        'align-self': 'center',
    })

dataset_upload_pop_up = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Confirm data types")),
        dbc.ModalBody(id="modal-body"),
        dbc.ModalFooter(
            [daq.BooleanSwitch(
                id='allow-kde',
                label='Allow KDE Factors',
                on=False,
                style={'fontWeight': 'bold'}
            ),
                dbc.Button("Confirm", id="save-selections", className="ms-auto", n_clicks=0)],
            style={"height": "100px"}),
    ],
    id="modal-body-scroll",
    centered=True,
    scrollable=True,
    is_open=False,
    keyboard=False,
    backdrop="static"
)
bn_export_pop_up = dbc.Modal(
    [
        dbc.ModalHeader(dbc.ModalTitle("Export structure as:")),
        dbc.ModalBody([

            dbc.Button('CSV (Adjacency Matrix)', id='bt-export-csv-m', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dbc.Button('CSV (Adjacency List)', id='bt-export-csv-l', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dbc.Button('CSV (Factor Types)', id='bt-export-csv-types', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dbc.Button('JPG', id='bt-export-jpg', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dbc.Button('PNG', id='bt-export-png', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dbc.Button('SVG', id='bt-export-svg', style={
                "margin": 20}, className="me-1", color="primary", outline=True),
            dcc.Download(id="download-item")
        ],
            className="d-flex flex-column align-items-center justify-content-center",),

    ],
    id="modal-download-dialogue",
    centered=True,
    scrollable=True,
    is_open=False,
)


pop_ups = html.Div(
    [dataset_upload_pop_up,
     bn_export_pop_up,
     dcc.Store(id="page-load", storage_type="memory")
     ]
)

operations_panel = html.Div(
    [operation_header,
        operation_selection,
        apply_operation_menu
     ])

right_panel = html.Div([select_learning_alg,
                        operations_panel,
                        arc_constraint_lists,
                        error_notification,
                        ],


                       style={"margin": 20, 'flex': 1})

left_panel = html.Div(
    [layout_menu,
        dcc.Interval(id='apply-loop', interval=1000,
                     n_intervals=0, disabled=True),
        cyto_board,
        loglik_graph
     ], style={"margin": 20, 'flex': 1, 'height': '0',
               'padding-bottom': '100%'})


cache = Cache(server)  # Initialize the cache with Flask


def serve_layout():
    layout = html.Div([
        # Sección superior: Título, links y texto
        html.Div([
            html.H1("Interactive Structural Learning", style={'textAlign': 'center'}),
            
            html.Div(
                className="link-bar",
                style={"textAlign": "center", "marginBottom": "20px"},
                children=[
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Original GitHub"
                        ],
                        href="https://github.com/JuanFPR-UPM/pybnesian_ainguraiiot",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/2023/11/cropped-logo_CIG.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Paper PDF"
                        ],
                        href="https://cig.fi.upm.es/jcr-articles/",
                        target="_blank",
                        className="btn btn-outline-primary me-2"
                    ),
                    html.A(
                        children=[
                            html.Img(
                                src="https://cig.fi.upm.es/wp-content/uploads/github.png",
                                style={"height": "24px", "marginRight": "8px"}
                            ),
                            "Dash Adapted GitHub"
                        ],
                        href="https://github.com/KeevinPR/pybnesian_ainguraiiot",
                        target="_blank",
                        className="btn btn-outline-info me-2"
                    ),
                ]
            ),
            
            html.Div(
                [
                    html.P(
                        "PyBNesian is a Python package that implements Bayesian networks. Currently, it is mainly dedicated to learning Bayesian networks."
                        "PyBNesian is implemented in C++, to achieve significant performance gains.",
                        style={"textAlign": "center", "maxWidth": "800px", "margin": "0 auto"}
                    )
                ],
                style={"marginBottom": "20px"}
            ),
        ], style={'font-size':'20px','width': '100%', 'textAlign': 'center'}),
        
        # Sección inferior: Paneles con estilo de filas
        html.Div(
            [left_panel, right_panel, pop_ups],
            style={'display': 'flex', 'flexDirection': 'row'}
        )
    ])
    return layout


app.layout = serve_layout
