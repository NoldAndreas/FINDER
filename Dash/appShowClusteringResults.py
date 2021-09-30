import json

import dash
import dash_table
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import numpy as np

import plotly.graph_objects as go

import sys
sys.path.append('../Code/')
from ClusterBasing import ClusterBasing
#from glob import glob
import os



#************************
#************************
#Parameters
#************************
#************************

Cells = ['Cell 1','Cell 2'];

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

basefolder          = "../../AnalysisDataOrganized/";
Cells = [o for o in os.listdir(basefolder) if os.path.isdir(os.path.join(basefolder,o))];

outputfolder        = basefolder+Cells[0]+"/Output/"; #"AHA_2_MMStack_Pos0.ome_locs_render_driftcor_filter_render_pix0.02X6f20/Output/"
Windows             = [o for o in os.listdir(outputfolder) if os.path.isdir(os.path.join(outputfolder,o))];
outputfolder_window = os.path.join(outputfolder,Windows[0]);
#os.path.join(outputfolder, o)

CB = ClusterBasing(outputfolder_window);
CB.GetClusterings_InOutCell();
CB.GetReferenceClustering();

marks_sigma = {};
for s in np.unique(CB.phasespace_all['sigma']):
    marks_sigma[s] = str(np.round(s,3));

marks_threshold = {};
for t in np.unique(CB.phasespace_all['threshold']):
    marks_threshold[t] = str(t);


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


#fig.update_traces(marker_size=10)

app.layout = html.Div([
    html.H1(children='Clustering Results'),

    html.Div([
        dcc.Dropdown(
            id='cell-option',
            options=[{'label': i, 'value': i} for i in Cells],
            value=Cells[0]
        ),
        dcc.Dropdown(
            id='window-option',
            options=[{'label': i, 'value': i} for i in Windows],
            value=Windows[0]
        ),
    ]),
    html.Br(),
    html.H2(children='Number of clusters'),
    html.Div([
        html.Div(
        dcc.Graph(
            id='graph_noCluster_incell'
        ),style={'width': '49%', 'display': 'inline-block'}),
        html.Div(
        dcc.Graph(
            id='graph_noCluster_outcell'
        ),style={'width': '49%', 'display': 'inline-block'}),
    ]),

    html.Div([
        html.Div(id='updatemode-infochosen', style={'margin-top': 20}),
        html.Div(
        dcc.Graph(
            id='graph_noCluster_distribution'
        ),style={'width': '49%', 'display': 'inline-block'}),
        html.Div(
        dcc.Graph(
            id='graph_scatter_incell_chosen'
        ),style={'width': '24%', 'display': 'inline-block'}),
        html.Div(
        dcc.Graph(
            id='graph_scatter_outcell_chosen'
        ),style={'width': '24%', 'display': 'inline-block'}),
    ]),

    html.Br(),
    html.H2(children='Number of clusters above threshold'),
    html.Div([
        html.Div(
        dcc.Graph(
            id='graph_noCluster_incell_aboveT'
        ),style={'width': '49%', 'display': 'inline-block'}),
        html.Div(
        dcc.Graph(
            id='graph_noCluster_outcell_aboveT'
        ),style={'width': '49%', 'display': 'inline-block'}),
    ]),

    html.Br(),
    html.H2(children='Specific clustering'),

    html.Div([
        dcc.Slider(
            id='my-slider-sigma',
            min=np.min(list(marks_sigma.keys())),
            max=np.max(list(marks_sigma.keys())),
            step=None,
            marks=marks_sigma,
            value=(list(marks_sigma.keys()))[0],
        ),
        html.Div(id='updatemode-output-sigma', style={'margin-top': 20}),
        dcc.Slider(
            id='my-slider-threshold',
            min=np.min(list(marks_threshold.keys())),
            max=np.max(list(marks_threshold.keys())),
            step=None,
            marks=marks_threshold,
            value=(list(marks_threshold.keys()))[0],
        ),
        html.Div(id='updatemode-output-threshold', style={'margin-top': 20}),
    ]),
    html.Div([
        html.Div(
        dcc.Graph(
            id='graph_scatter_incell'
        ),style={'width': '49%', 'display': 'inline-block'}),
        html.Div(
        dcc.Graph(
            id='graph_scatter_outcell'
        ),style={'width': '49%', 'display': 'inline-block'}),
    ]),
])


@app.callback(
    dash.dependencies.Output('updatemode-output-sigma', 'children'),
    dash.dependencies.Output('updatemode-output-threshold', 'children'),
    dash.dependencies.Output('graph_noCluster_incell', 'figure'),
    dash.dependencies.Output('graph_noCluster_outcell', 'figure'),
    dash.dependencies.Output('updatemode-infochosen', 'children'),
    dash.dependencies.Output('graph_noCluster_distribution', 'figure'),
    dash.dependencies.Output('graph_scatter_incell_chosen', 'figure'),
    dash.dependencies.Output('graph_scatter_outcell_chosen', 'figure'),
    dash.dependencies.Output('graph_noCluster_incell_aboveT', 'figure'),
    dash.dependencies.Output('graph_noCluster_outcell_aboveT', 'figure'),
    dash.dependencies.Output('graph_scatter_incell', 'figure'),
    dash.dependencies.Output('graph_scatter_outcell', 'figure'),
    #dash.dependencies.Input('button', 'n_clicks'),
    dash.dependencies.Input('my-slider-sigma','value'),
    dash.dependencies.Input('my-slider-threshold','value'),
    dash.dependencies.Input('cell-option', 'value'),
    dash.dependencies.Input('window-option', 'value'),
    )
def update_output(value_sigma,value_threshold,value_cell,value_window):
    #df = pd.DataFrame();
    #df['x'] = np.arange(10);
    #df['Value'] = np.ones_like(np.arange(10));
#    fig = px.line(df, x="x", y="Value",title="Protein and mRNA concentrations (last computation):");
#    fig2 = px.line(df, x="x", y="Value",title="Protein and mRNA concentrations (last computation):");

    changed_id = [p['prop_id'] for p in dash.callback_context.triggered][0];

    global CB,outputfolder_window

    if(('cell-option' in changed_id) or ('window-option' in changed_id)):
        outputfolder        = basefolder+value_cell+"/Output/";
        outputfolder_window = os.path.join(outputfolder,value_window);

        CB = ClusterBasing(outputfolder_window);
        CB.GetClusterings_InOutCell();
        CB.GetReferenceClustering();


    #*************************************************
    # Number of clusters
    #*************************************************
    heatmap1_data = pd.pivot_table(CB.phasespace_all, values='no_clusters',
                         index=['threshold'],
                         columns='sigma')
    fig1 = go.Figure(data =go.Heatmap(z=heatmap1_data,x=heatmap1_data.columns,y=heatmap1_data.index))

    heatmap1_data = pd.pivot_table(CB.phasespace_all, values='no_clusters_ref',
                         index=['threshold'],
                         columns='sigma')
    fig2 = go.Figure(data =go.Heatmap(z=heatmap1_data,x=heatmap1_data.columns,y=heatmap1_data.index))

    #*************************************************
    # Distribution of number of clusters
    #*************************************************
    df = px.data.tips()
    fig_nocl_dist = px.histogram(CB.df_clusterSizes_all, x="clusterSize", color="type", marginal="rug",hover_data=CB.df_clusterSizes_all.columns,\
                      histnorm='probability density')

    chosen_row = CB.GetClustering();
    text_infochosen = 'T = '+str(chosen_row['T'])+' , sigma = '+str(chosen_row['sigma'])+' , threshold = '+str(chosen_row['threshold']);


    labels_ = chosen_row['labels'];
    markNoise = (labels_==-1);
    figchosen_incell = go.Figure()
    figchosen_incell.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise,0], y=CB.XC_incell[markNoise,1],
                   marker=dict(color='Gray',size=2)));
    figchosen_incell.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise==False,0], y=CB.XC_incell[markNoise==False,1],
                              marker=dict(color='Red',size=4)))
    figchosen_incell.update_layout(width=500,height=500);

    labels_ = chosen_row['labels_ref'];
    markNoise = (labels_==-1);
    figchosen_outcell = go.Figure()
    figchosen_outcell.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise,0], y=CB.XC_outcell[markNoise,1],
                   marker=dict(color='Gray',size=2)));
    figchosen_outcell.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise==False,0], y=CB.XC_outcell[markNoise==False,1],
                              marker=dict(color='Red',size=4)))
    figchosen_outcell.update_layout(width=500,height=500)

    #*************************************************
    # Number of clusters above threshold
    #*************************************************
    heatmap1_data = pd.pivot_table(CB.phasespace_all_aboveT, values='no_clusters',
                         index=['threshold'],
                         columns='sigma')
    fig3 = go.Figure(data =go.Heatmap(z=heatmap1_data,x=heatmap1_data.columns,y=heatmap1_data.index))

    heatmap1_data = pd.pivot_table(CB.phasespace_all_aboveT, values='no_clusters_ref',
                         index=['threshold'],
                         columns='sigma')
    fig4 = go.Figure(data =go.Heatmap(z=heatmap1_data,x=heatmap1_data.columns,y=heatmap1_data.index))

    #*************************************************
    # Scatterplot
    #*************************************************
    idx = np.argmax((CB.phasespace_all['sigma']==value_sigma) & \
        (CB.phasespace_all['threshold']==value_threshold));

    text_sigma      = "Sigma = "+str(value_sigma) + ' idx '+str(idx);
    text_threshold  = "Threshold = "+str(value_threshold);

    labels_ = CB.phasespace_all.loc[idx,'labels'];
    markNoise = (labels_==-1);
    #labels_[labels_>=0] = 0;
    fig5 = go.Figure()

    fig5.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise,0], y=CB.XC_incell[markNoise,1],
                   marker=dict(color='Gray',size=2)));
    fig5.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise==False,0], y=CB.XC_incell[markNoise==False,1],
                              marker=dict(color='Red',size=4)))
    fig5.update_layout(width=500,height=500)

    #fig5.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise,0], y=CB.XC_incell[markNoise,1]));
    #fig5.add_trace(go.Scatter(mode='markers',x=CB.XC_incell[markNoise==False,0], y=CB.XC_incell[markNoise==False,1]))
    #fig3.show();
    #fig3 = px.scatter(x=CB.XC_incell[:,0], y=CB.XC_incell[:,1],color=labels_);

    labels_ = CB.phasespace_all.loc[idx,'labels_ref'];
    markNoise = (labels_==-1);
    #labels_[labels_>=0] = 0;
    fig6 = go.Figure()
    fig6.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise,0], y=CB.XC_outcell[markNoise,1],
                   marker=dict(color='Gray',size=2)));
    fig6.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise==False,0], y=CB.XC_outcell[markNoise==False,1],
                              marker=dict(color='Red',size=4)))
    fig6.update_layout(width=500,height=500)

    #fig6.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise,0], y=CB.XC_outcell[markNoise,1]));
    #fig6.add_trace(go.Scatter(mode='markers',x=CB.XC_outcell[markNoise==False,0], y=CB.XC_outcell[markNoise==False,1]))


#    fig6 = px.scatter(x=CB.XC_outcell[:,0], y=CB.XC_outcell[:,1])

    return text_sigma,text_threshold,fig1,fig2,text_infochosen,fig_nocl_dist,figchosen_incell,figchosen_outcell,fig3,fig4,fig5,fig6


if __name__ == '__main__':
    app.run_server(debug=True)
