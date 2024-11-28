import logging
from pathlib import Path
from typing import Optional
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from torch_geometric.data import HeteroData

from anemoi.graphs.plotting.prepare import compute_isolated_nodes
from anemoi.graphs.plotting.style import *

LOGGER = logging.getLogger(__name__)


def plot_isolated_nodes(graph: HeteroData, out_file: Optional[Union[str, Path]] = None) -> None:
    """Plot isolated nodes.

    This method creates an interactive visualization of the isolated nodes in the graph.

    Parameters
    ----------
    graph : AnemoiGraph
        The graph to plot.
    out_file : str | Path, optional
        Name of the file to save the plot. Default is None.
    """
    isolated_nodes = compute_isolated_nodes(graph)

    if len(isolated_nodes) == 0:
        LOGGER.warning("No isolated nodes found.")
        return

    colorbar = plt.cm.rainbow(np.linspace(0, 1, len(isolated_nodes)))
    nodes = []
    for name, (lat, lon) in isolated_nodes.items():
        nodes.append(
            go.Scattergeo(
                lat=lat,
                lon=lon,
                mode="markers",
                hoverinfo="text",
                name=name,
                marker={"showscale": False, "color": colorbar[len(nodes)], "size": 10},
            ),
        )

    layout = go.Layout(
        title="<br>Orphan nodes",
        titlefont_size=16,
        showlegend=True,
        hovermode="closest",
        margin={"b": 20, "l": 5, "r": 5, "t": 40},
        annotations=[annotations_style],
        legend={"x": 0, "y": 1},
        xaxis=plotly_axis_config,
        yaxis=plotly_axis_config,
    )
    fig = go.Figure(data=nodes, layout=layout)
    fig.update_geos(fitbounds="locations")

    if out_file is not None:
        fig.write_html(out_file)
    else:
        fig.show()
