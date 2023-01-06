import sys

import colorlover as cl
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import scipy.stats as stats
import utils.cell_segmentation as cell
from sklearn import metrics

sys.path.append("../../Modules/")
from ClustersInOutCell import ClustersInOutCell


def make_default_figure():
    layout = go.Layout(
        title="Plot 2 (Statistic along line of optima)",
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
        xaxis=dict(title="minPts"),
    )

    figure = go.Figure(layout=layout)
    return figure


def serve_scatterplot(xc, labels=None):

    layout = go.Layout(
        title="Localizations (Plot 3)",
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
        showlegend=False,
        xaxis={"showgrid": False},
        yaxis={"showgrid": False},
    )

    figure = go.Figure(layout=layout)

    if labels is None:
        figure.add_trace(
            go.Scatter(
                mode="markers",
                x=xc[:, 0],
                y=xc[:, 1],
                marker=dict(color="Gray", size=2),
            )
        )
    else:
        figure.add_trace(
            go.Scatter(
                mode="markers",
                x=xc[labels == -1, 0],
                y=xc[labels == -1, 1],
                marker=dict(color="Gray", size=2),
            )
        )

        fig_markers = px.scatter(
            x=xc[labels >= 0, 0],
            y=xc[labels >= 0, 1],
            color=(labels[labels >= 0]).astype(str),
        )
        for d in fig_markers.data:
            figure.add_trace(d)
        # figure.add_trace(go.Scatter(mode='markers',x=xc[labels>=0,0], y=xc[labels>=0,1],
    #                   marker=dict(color=labels.astype(str),size=2)))

    figure.update_yaxes(
        autorange="reversed", scaleanchor="x", constrain="domain"
    )
    figure.update_xaxes(constrain="domain")

    return figure


def serve_along_optima_figure(cb, tuple_statistics, change_threshold=0.95):

    key = tuple_statistics[0]

    df_ = cb.df_clusters_opt_th.loc[cb.df_clusters_opt_th.type == "incell", :]

    df_grouped = df_.groupby(["threshold"])["clusterSize"].agg(
        [tuple_statistics]
    )
    value = df_grouped[key]

    # Rescale and find first time 90% change is detected
    x_values = df_grouped.index
    rel_value = np.asarray(
        (value - np.min(value)) / (np.max(value) - np.min(value))
    )
    if rel_value[0] > rel_value[-1]:
        rel_value = 1 - rel_value

    idx = np.where(rel_value > change_threshold)[0][0]

    layout = go.Layout(
        title="Plot 2 (Statistic along line of optima)",
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
        xaxis=dict(title="minPts"),
    )

    figure = go.Figure(layout=layout)

    figure.add_trace(
        go.Scatter(x=x_values, y=value, mode="lines+markers", name=key)
    )

    figure.add_vline(
        x=x_values[idx], line_width=3, line_dash="dash", line_color="red"
    )

    return figure


def serve_phasespace_figure(cb, quantity="no_clusters"):

    layout = go.Layout(
        title="Plot 1 (selected quantity in phasespace)",
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
        xaxis=dict(title="sigma", type="log"),
        yaxis=dict(title="minPts", type="log"),
    )

    figure = go.Figure(layout=layout)

    heatmap1_data = pd.pivot_table(
        cb.phasespace_all,
        values=quantity,
        index=["threshold"],
        columns="sigma",
    )
    figure.add_trace(
        go.Heatmap(
            z=heatmap1_data, x=heatmap1_data.columns, y=heatmap1_data.index
        )
    )

    # Plot line of optima
    figure.add_trace(
        go.Scatter(
            x=cb.df_opt_th["sigma"],
            y=cb.df_opt_th["threshold"],
            mode="lines+markers",
            name="Line of optima",
        )
    )

    return figure


def serve_histogram(cell_sample):

    h = cell_sample.cutoff_image.flatten()

    layout = go.Layout(
        title="Histogram",
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    figure = go.Figure(layout=layout)

    figure.add_histogram(x=h[h > 0])
    # fig = px.histogram(data)

    figure.add_vline(
        x=cell_sample.threshold_otsu,
        line_width=3,
        line_dash="dash",
        line_color="red",
    )

    return figure


def serve_overview_plot(h, title):
    # data = [trace0, trace1, trace2, trace3]
    # figure = go.Figure(data=data, layout=layout)

    #    data = [px.imshow(h)]

    layout = go.Layout(
        title=title,
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    figure = go.Figure(layout=layout)

    figure.update_yaxes(
        autorange="reversed", scaleanchor="x", constrain="domain"
    )
    figure.update_xaxes(constrain="domain")

    figure.add_trace(go.Heatmap(z=h))
    return figure


def serve_prediction_plot(
    model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
):
    # Get train and test score from model
    y_pred_train = (model.decision_function(X_train) > threshold).astype(int)
    y_pred_test = (model.decision_function(X_test) > threshold).astype(int)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(
        abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max())
    )

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    # Plot the threshold
    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False,
            type="constraint",
            operation="=",
            value=scaled_threshold,
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    # Plot Training Data
    trace2 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale),
    )

    # Plot Test Data
    trace3 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(
            size=10,
            symbol="triangle-up",
            color=y_test,
            colorscale=bright_cscale,
        ),
    )

    layout = go.Layout(
        xaxis=dict(
            ticks="", showticklabels=False, showgrid=False, zeroline=False
        ),
        yaxis=dict(
            ticks="", showticklabels=False, showgrid=False, zeroline=False
        ),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0, trace1, trace2, trace3]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_roc_curve(model, X_test, y_test):
    decision_test = model.decision_function(X_test)
    fpr, tpr, threshold = metrics.roc_curve(y_test, decision_test)

    # AUC Score
    auc_score = metrics.roc_auc_score(y_true=y_test, y_score=decision_test)

    trace0 = go.Scatter(
        x=fpr,
        y=tpr,
        mode="lines",
        name="Test Data",
        marker={"color": "#13c6e9"},
    )

    layout = go.Layout(
        title=f"ROC Curve (AUC = {auc_score:.3f})",
        xaxis=dict(title="False Positive Rate", gridcolor="#2f3445"),
        yaxis=dict(title="True Positive Rate", gridcolor="#2f3445"),
        legend=dict(x=0, y=1.05, orientation="h"),
        margin=dict(l=100, r=10, t=25, b=40),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_pie_confusion_matrix(model, X_test, y_test, Z, threshold):
    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    y_pred_test = (model.decision_function(X_test) > scaled_threshold).astype(
        int
    )

    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    tn, fp, fn, tp = matrix.ravel()

    values = [tp, fn, fp, tn]
    label_text = [
        "True Positive",
        "False Negative",
        "False Positive",
        "True Negative",
    ]
    labels = ["TP", "FN", "FP", "TN"]
    blue = cl.flipper()["seq"]["9"]["Blues"]
    red = cl.flipper()["seq"]["9"]["Reds"]
    colors = ["#13c6e9", blue[1], "#ff916d", "#ff744c"]

    trace0 = go.Pie(
        labels=label_text,
        values=values,
        hoverinfo="label+value+percent",
        textinfo="text+value",
        text=labels,
        sort=False,
        marker=dict(colors=colors),
        insidetextfont={"color": "white"},
        rotation=90,
    )

    layout = go.Layout(
        title="Confusion Matrix",
        margin=dict(l=50, r=50, t=100, b=10),
        legend=dict(
            bgcolor="#282b38", font={"color": "#a5b1cd"}, orientation="h"
        ),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0]
    figure = go.Figure(data=data, layout=layout)

    return figure
