import time
import importlib
import os

import dash
import dash_core_components as dcc
import dash_html_components as html
import matplotlib as matplotlib
from matplotlib.pyplot import figure, text
import numpy as np
from dash.dependencies import Input, Output, State
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
import scipy.stats as stats

import utils.dash_reusable_components as drc
import utils.figures as figs

#import utils.cell_segmentation as cell

import sys
sys.path.append('../../Modules/')
from ClustersInOutCell import ClustersInOutCell

matplotlib.pyplot.switch_backend('Agg') 

BASEDIR = "../../../Data_AnalysisOrganized/" 
DIRS = sorted((f for f in os.listdir(BASEDIR) if not f.startswith(".")), key=str.lower)

STATISTICS = [('median',np.median),\
            ('count','count'),\
            ('min','min'),\
            ('max','max'),\
            ('std','std'),\
            ('cv',stats.variation),\
            ('skewness',stats.skew),\
            ('kurtosis',stats.kurtosis),\
            ('fano',lambda d_ : np.var(d_)/np.mean(d_))]            
 

app = dash.Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Cluster Visualization"
server = app.server


app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Cluster Visualization",
                                    href="https://google.com",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                        html.A(
                            id="banner-logo",
                            children=[
                                html.Img(src=app.get_asset_url("dash-logo-new.png"))
                            ],
                            href="https://plot.ly/products/dash/",
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            style={'width': '20%'},
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options = [
                                                {'label': i, 'value': i} for i in DIRS
                                            ],                                            
                                            clearable=False,
                                            searchable=False,
                                            value=DIRS[0],
                                        ),
                                        drc.NamedDropdown(
                                            name="Select Window",
                                            id="dropdown-select-window",
                                            clearable=False,
                                            searchable=False,                                            
                                        ),
                                    ]
                                ),
                                drc.Card(
                                    id="second-card",
                                    children=[
                                       drc.NamedRadioItems(
                                           name="Quantity (plot 1)",
                                           id="phasespace-quantity",
                                           options=[
                                               {'label':'Number of clusters','value':'no_clusters'},
                                               {'label':'Similarity score','value':'similarityScore'}],
                                            value='no_clusters'
                                        ),                                    
                                        drc.NamedRadioItems(
                                           name="Statistic along ling of optima (plot 2)",
                                           id="along-optima-quantity",
                                           options=[
                                               {'label':t[0],'value':t[0]} for t in STATISTICS
                                            ],
                                            value='median'
                                       ),
                                       drc.NamedSlider(
                                            name="Percent threshold to set minPts (red dashed line)",
                                            id="slider-change-threshold",
                                            min=0.5,
                                            max=1,
                                            step=0.05,                                            
                                            value=0.9,
                                            marks = {str(i): str(i) for i in [0.5,1]}
                                        ),
                                       drc.NamedRadioItems(
                                           name="Show clusters (plot 3)",
                                           id="show-clusters",
                                           options=[
                                               {'label':'Yes','value':1},
                                               {'label':'No','value':0},
                                            ],
                                            value=1
                                       )                                    
                                    ]
                                )                           
                            ],
                        ),
                        html.Div(
                            id="div-phasespace",
                            style={'width': '30%'},
                            children=dcc.Graph(
                                id="graph-phasespace",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),                        
                        html.Div(
                            id="div-scatter",
                            style={'width': '49%'},
                            children=dcc.Graph(
                                id="graph-scatter",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),                        
                    ],
                )
            ],
        ),
    ]
)


@app.callback(Output("div-scatter", "children"),
              [Input('graph-phasespace','clickData'),
               Input('show-clusters','value'),
               Input('dropdown-select-window', 'value')],               
              [State("dropdown-select-dataset","value")])
def update_scatterplot(clickData, show_clusters, window, dataset):
    
    if dataset and window:
        
        outputfolder_window = os.path.join(BASEDIR,dataset,'Output',window)
        
        CB = ClustersInOutCell(outputfolder_window)
        CB.GetClusterings_InOutCell()        
        
        if clickData and show_clusters:

            node_sigma = clickData['points'][0]['x']
            node_threshold = clickData['points'][0]['y']

            ps = CB.get_info_phasespace_approx(node_sigma,node_threshold)
            figure_scatterplot = figs.serve_scatterplot(CB.XC_incell,ps['labels'])
        else:
            figure_scatterplot = figs.serve_scatterplot(CB.XC_incell)

        return  html.Div(
                    id="svm-graph-container",
                    children=[
                        dcc.Loading(
                            className="graph-wrapper1",
                            children=dcc.Graph(id="graph-scatter", figure=figure_scatterplot),
                            style={"display": "none"},
                        ),                                                      
                    ]
                )


@app.callback(
     Output('dropdown-select-window', 'options'),
    [Input('dropdown-select-dataset', 'value')]
)
def update_window_selection(dataset):
    output_dir = os.path.join(BASEDIR,dataset,"Output")
    if os.path.isdir(output_dir):
        list_ = [d for d in os.listdir(output_dir) if os.path.isdir(os.path.join(output_dir,d))]
    else:
        list_ = []
    
    return [{'value' : i, 'label' : i} for i in list_]    

@app.callback(    
    Output("div-phasespace", "children"),
    [                
        Input("dropdown-select-dataset","value"),
        Input('dropdown-select-window', 'value'),
        Input('phasespace-quantity', 'value'),
        Input('along-optima-quantity', 'value'),
        Input('slider-change-threshold', 'value')     
    ],
)
def update_svm_graph(  
    dataset,
    window,
    quantity,
    quantity_statistics,
    change_threshold
):    
    if dataset and window:
        outputfolder_window = os.path.join(BASEDIR,dataset,'Output',window)

        CB = ClustersInOutCell(outputfolder_window)
        CB.GetClusterings_InOutCell()
        CB.GetSimilarityAlongOptima() 
        
        figure_phasespace = figs.serve_phasespace_figure(CB,quantity)   

        #STATISTICS    
        l = [t for t in STATISTICS if t[0] == quantity_statistics]
        figure_along_optima = figs.serve_along_optima_figure(CB, l[0], change_threshold=change_threshold)
    
        return  html.Div(
                    id="svm-graph-container",
                    children=[
                        dcc.Loading(
                            className="graph-wrapper",
                            children=dcc.Graph(id="graph-phasespace", figure=figure_phasespace),
                            style={"display": "none"},
                        ),
                        dcc.Loading(
                            className="graph-wrapper2",
                            children=dcc.Graph(id="graph-along-optima", figure=figure_along_optima),
                            style={"display": "none"},
                        ),
                    ]
                )               
    else:
        return []


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
