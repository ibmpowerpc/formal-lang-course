from pathlib import Path

import cfpq_data
import pytest
from networkx import MultiDiGraph

from project.__main__ import build_two_cycled_graph, count_by_name


def test_count_by_name_returns_graph_statistics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    graph_path = tmp_path / "example.csv"
    graph = MultiDiGraph()
    graph.add_edge(0, 1, label="first")
    graph.add_edge(1, 2, label="rest")
    graph.add_edge(2, 0, label="first")

    def download(name: str) -> Path:
        assert name == "example"
        return graph_path

    def graph_from_csv(path: Path) -> MultiDiGraph:
        assert path == graph_path
        return graph

    monkeypatch.setattr(cfpq_data, "download", download)
    monkeypatch.setattr(cfpq_data, "graph_from_csv", graph_from_csv)

    nodes_count, edges_count, labels = count_by_name("example")

    assert nodes_count == 3
    assert edges_count == 3
    assert labels == {"first", "rest"}


def test_count_by_name_propagates_download_error(
    monkeypatch: pytest.MonkeyPatch,
):
    def download(name: str) -> Path:
        raise FileNotFoundError(f"No graph with name={name!r} found")

    monkeypatch.setattr(cfpq_data, "download", download)

    with pytest.raises(FileNotFoundError, match="unknown"):
        count_by_name("unknown")


def test_build_two_cycled_graph_returns_expected_graph():
    first_cycle_size = 2
    second_cycle_size = 3

    graph = build_two_cycled_graph(
        first_cycle_size,
        second_cycle_size,
        labels=("left", "right"),
    )

    assert isinstance(graph, MultiDiGraph)
    assert graph.number_of_nodes() == first_cycle_size + second_cycle_size + 1
    assert graph.number_of_edges() == first_cycle_size + second_cycle_size + 2

    labels = [data["label"] for _, _, data in graph.edges(data=True)]
    assert labels.count("left") == first_cycle_size + 1
    assert labels.count("right") == second_cycle_size + 1


def test_build_two_cycled_graph_uses_zero_as_common_node():
    graph = build_two_cycled_graph(2, 3, labels=("left", "right"))

    outgoing_labels = {
        data["label"] for _, _, data in graph.out_edges(0, data=True)
    }
    incoming_labels = {
        data["label"] for _, _, data in graph.in_edges(0, data=True)
    }

    assert outgoing_labels == {"left", "right"}
    assert incoming_labels == {"left", "right"}
