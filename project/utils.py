from pathlib import Path

import cfpq_data
import pydot
from networkx import MultiDiGraph


def count_by_name(name: str):
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        {data["label"] for _, _, data in graph.edges(data=True)},
    )


def build_two_cycled_graph(
    n: int,
    m: int,
    labels: tuple[str, str],
    output_path: str | Path,
) -> MultiDiGraph:
    graph = cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)

    dot_graph = pydot.Dot("two_cycles", graph_type="digraph")
    for node in graph.nodes:
        dot_graph.add_node(pydot.Node(str(node)))
    for source, target, data in graph.edges(data=True):
        dot_graph.add_edge(
            pydot.Edge(str(source), str(target), label=data["label"])
        )

    dot_graph.write_raw(str(output_path))
    return graph
