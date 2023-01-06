import base64
import datetime
import io
import os
import sys

import dash
import matplotlib as matplotlib
import numpy as np
import pandas as pd
from dash import dash_table, dcc, html
from dash.dependencies import Input, Output, State
from matplotlib.pyplot import figure, text

current = os.path.dirname(os.path.realpath(__file__))

# Getting the parent directory name
# where the current directory is present.
parent = os.path.dirname(current)

modules = os.path.join(parent, "Modules")

# adding the parent directory to
# the sys.path.
sys.path.append(modules)


from Finder import Finder

external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"]

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(
    [
        dcc.Upload(
            id="upload-data",
            children=html.Div(["Drag and Drop or ", html.A("Select Files")]),
            style={
                "width": "100%",
                "height": "60px",
                "lineHeight": "60px",
                "borderWidth": "1px",
                "borderStyle": "dashed",
                "borderRadius": "5px",
                "textAlign": "center",
                "margin": "10px",
            },
        ),
        html.Div(id="name-of-file", children="no file selected"),
        html.Div(
            [
                html.Div(
                    [
                        "points per dimension",
                        dcc.Input(value=15, id="points-per-dimension"),
                    ]
                ),
                html.Div(
                    [
                        "algorithm",
                        dcc.Dropdown(
                            ["DbscanLoop", "dbscan"],
                            value="DbscanLoop",
                            id="algorithm",
                        ),
                    ]
                ),
                html.Div(
                    [
                        "minimun threshold",
                        dcc.Input(value=5, id="minimum-threshold"),
                    ]
                ),
                html.Div(
                    [
                        "maximum threshold",
                        dcc.Input(value=20, id="maximum-threshold"),
                    ]
                ),
                html.Div(
                    [
                        "threshold values",
                        dcc.RadioItems(
                            ["Linear", "Log"],
                            value="Linear",
                            id="threshold-values",
                        ),
                    ]
                ),
                html.Div(
                    [
                        "sigma values",
                        dcc.RadioItems(
                            ["Linear", "Log"], value="Log", id="sigma-values"
                        ),
                    ]
                ),
                html.Button(
                    "Run FINDER and download labels",
                    id="run-FINDER",
                    n_clicks=0,
                ),
                dcc.Download(id="download-dataframe-csv"),
            ]
        ),
    ]
)


@app.callback(
    Output("download-dataframe-csv", "data"),
    Input("run-FINDER", "n_clicks"),
    State("upload-data", "contents"),
    State("upload-data", "filename"),
    State("points-per-dimension", "value"),
    State("algorithm", "value"),
    State("minimum-threshold", "value"),
    State("maximum-threshold", "value"),
    State("threshold-values", "value"),
    State("sigma-values", "value"),
)
def run_FINDER(
    n_clicks,
    contents,
    filename,
    points_per_dimension,
    algorithm,
    min_th,
    max_th,
    th_vals,
    sigma_vals,
):
    if n_clicks:

        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        df = pd.read_csv(
            io.StringIO(decoded.decode("utf-8")), names=["x", "y"], sep=" "
        )
        X = df.values

        minmax_th = [min_th, max_th]
        log_thresholds = True
        if th_vals == "Log":
            log_threshold = False
        log_sigmas = True
        if sigma_vals == "Log":
            log_sigmas = False

        model = Finder(
            similarity_score_computation="threshold",
            points_per_dimension=points_per_dimension,
            algo=algorithm,
            minmax_threshold=minmax_th,
            log_thresholds=log_thresholds,
            log_sigmas=log_sigmas,
        )
        model.fit(X)
        labels = model.labels

        df = pd.DataFrame({"x": X[:, 0], "y": X[:, 1], "label": labels})
        return dcc.send_data_frame(df.to_csv, filename + "_labels.csv")


@app.callback(
    Output("name-of-file", "children"), Input("upload-data", "filename")
)
def display_file(filename):
    if filename:
        return "file selected: " + filename


# @app.callback(
#     Output("download-dataframe-csv", "data"),
#     Input("btn_csv", "n_clicks"),
#     State('download-dataframe-csv', 'value'),
# )
# def func(n_clicks, df):
#     if n_clicks:
#         #return dcc.send_data_frame(df.to_csv, "mydf.csv")
#
# @app.callback(
#     Output("download-dataframe-csv", "data"),
#     Input("btn_csv", "n_clicks"),
#     prevent_initial_call=True,
# )
# def func(n_clicks):
#     return dcc.send_data_frame(df.to_csv, "mydf.csv")

if __name__ == "__main__":

    app.run_server(debug=True)
