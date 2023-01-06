import importlib
import os
import time

import dash
import matplotlib as matplotlib
import numpy as np
import utils.cell_segmentation as cell
import utils.dash_reusable_components as drc
import utils.figures as figs
from dash import dcc, html
from dash.dependencies import Input, Output, State
from matplotlib.pyplot import figure, text
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

matplotlib.pyplot.switch_backend("Agg")

BASEDIR = "../../../Data_AnalysisOrganized/"

DIRS = sorted(
    (f for f in os.listdir(BASEDIR) if not f.startswith(".")), key=str.lower
)


app = dash.Dash(
    __name__,
    meta_tags=[
        {
            "name": "viewport",
            "content": "width=device-width, initial-scale=1.0",
        }
    ],
)
app.title = "Support Vector Machine"
server = app.server


def generate_data(n_samples, dataset, noise):
    if dataset == "moons":
        return datasets.make_moons(
            n_samples=n_samples, noise=noise, random_state=0
        )

    elif dataset == "circles":
        return datasets.make_circles(
            n_samples=n_samples, noise=noise, factor=0.5, random_state=1
        )

    elif dataset == "linear":
        X, y = datasets.make_classification(
            n_samples=n_samples,
            n_features=2,
            n_redundant=0,
            n_informative=2,
            random_state=2,
            n_clusters_per_class=1,
        )

        rng = np.random.RandomState(2)
        X += noise * rng.uniform(size=X.shape)
        linearly_separable = (X, y)

        return linearly_separable

    else:
        raise ValueError(
            "Data type incorrectly specified. Please choose an existing dataset."
        )


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
                                    "Split cell",
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
                                html.Img(
                                    src=app.get_asset_url("dash-logo-new.png")
                                )
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
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Select Dataset",
                                            id="dropdown-select-dataset",
                                            options=[
                                                {"label": i, "value": i}
                                                for i in DIRS
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
                                    ],
                                ),
                                drc.Card(
                                    id="second-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Number of pixels in x",
                                            id="slider-number-pixels",
                                            min=200,
                                            max=1000,
                                            step=200,
                                            marks={
                                                str(i): str(i)
                                                for i in [
                                                    200,
                                                    400,
                                                    600,
                                                    800,
                                                    1000,
                                                ]
                                            },
                                            value=800,
                                        ),
                                        drc.NamedSlider(
                                            name="Quantile Intensity Cutoff",
                                            id="slider-intensity-cutoff",
                                            min=0,
                                            max=1,
                                            step=0.2,
                                            marks={
                                                str(i): str(i)
                                                for i in [
                                                    0,
                                                    0.2,
                                                    0.4,
                                                    0.6,
                                                    0.8,
                                                    1,
                                                ]
                                            },
                                            value=0.8,
                                        ),
                                        drc.NamedSlider(
                                            name="Sigma for Gaussian filter",
                                            id="slider-gaussian-sigma",
                                            min=2,
                                            max=20,
                                            step=2,
                                            marks={
                                                str(i): str(i)
                                                for i in np.arange(2, 22, 2)
                                            },
                                            value=10,
                                        ),
                                        drc.NamedSlider(
                                            name="Padding at cell border",
                                            id="slider-cell-padding",
                                            min=0,
                                            max=50,
                                            step=10,
                                            marks={
                                                str(i): str(i)
                                                for i in [
                                                    0,
                                                    10,
                                                    20,
                                                    30,
                                                    40,
                                                    50,
                                                ]
                                            },
                                            value=20,
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="save-card",
                                    children=[
                                        html.Button(
                                            "Apply split",
                                            id="button-apply",
                                        ),
                                        dcc.Textarea(
                                            id="display-parameters",
                                            value="",
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38",
                                        paper_bgcolor="#282b38",
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


@app.callback(
    Output("display-parameters", "value"),
    [Input("button-apply", "n_clicks")],
    [
        State("dropdown-select-dataset", "value"),
        State("slider-number-pixels", "value"),
        State("slider-intensity-cutoff", "value"),
        State("slider-gaussian-sigma", "value"),
        State("slider-cell-padding", "value"),
    ],
)
def apply_split(
    n_clicks,
    dataset,
    number_pixels,
    intensity_cutoff,
    sigma_gaussian,
    cell_padding,
):
    if n_clicks:
        cell_name = os.path.join(BASEDIR, dataset)
        cell_sample = cell.CellSegmentation(cell_name)
        cell_sample.segmentation()

        cell_sample.parameters["N_x"] = number_pixels
        cell_sample.parameters["intensity_quantile_cutoff"] = intensity_cutoff
        cell_sample.parameters["sigma_gaussian_filter"] = sigma_gaussian
        cell_sample.parameters["pad_cell_border"] = cell_padding

        cell_sample.segmentation()

        cell_sample.save_split()
        text_output = "Saved split"

    else:
        text_output = ""

    return text_output


@app.callback(
    Output("dropdown-select-window", "options"),
    [Input("dropdown-select-dataset", "value")],
)
def update_window_selection(dataset):
    output_dir = os.path.join(
        "../../../AnalysisDataOrganized/", dataset, "Output"
    )
    if os.path.isdir(output_dir):
        list_ = [
            d
            for d in os.listdir(output_dir)
            if os.path.isdir(os.path.join(output_dir, d))
        ]
    else:
        list_ = []

    return [{"value": i, "label": i} for i in list_]


@app.callback(
    Output("div-graphs", "children"),
    [
        Input("dropdown-select-dataset", "value"),
        Input("slider-number-pixels", "value"),
        Input("slider-intensity-cutoff", "value"),
        Input("slider-gaussian-sigma", "value"),
        Input("slider-cell-padding", "value"),
    ],
)
def update_svm_graph(
    dataset, number_pixels, intensity_cutoff, sigma_gaussian, cell_padding
):
    cell_name = os.path.join(BASEDIR, dataset)
    cell_sample = cell.CellSegmentation(cell_name)
    cell_sample.segmentation()

    cell_sample.parameters["N_x"] = number_pixels
    cell_sample.parameters["intensity_quantile_cutoff"] = intensity_cutoff
    cell_sample.parameters["sigma_gaussian_filter"] = sigma_gaussian
    cell_sample.parameters["pad_cell_border"] = cell_padding

    cell_sample.segmentation()
    prediction_figure = figs.serve_overview_plot(
        cell_sample.cutoff_image, title="Input image after cutoff"
    )
    segmentation_figure_incell = figs.serve_overview_plot(
        cell_sample.im_incell.astype("int"), title="Incell area"
    )
    segmentation_figure_outcell = figs.serve_overview_plot(
        cell_sample.im_outcell.astype("int"), title="Outcell area"
    )

    histogram_figure = figs.serve_histogram(cell_sample)

    return [
        html.Div(
            id="svm-graph-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-sklearn-svm", figure=prediction_figure
                    ),
                    style={"display": "none"},
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-sklearn-svm", figure=histogram_figure
                    ),
                    style={"display": "none"},
                ),
            ],
        ),
        html.Div(
            id="graphs-container",
            children=[
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-sklearn-svm",
                        figure=segmentation_figure_incell,
                    ),
                    style={"display": "none"},
                ),
                dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(
                        id="graph-sklearn-svm",
                        figure=segmentation_figure_outcell,
                    ),
                    style={"display": "none"},
                ),
            ],
        ),
    ]


# Running the server
if __name__ == "__main__":
    app.run_server(debug=True)
