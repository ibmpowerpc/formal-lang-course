from pathlib import Path

import cfpq_data
import pydot
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


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
        dot_graph.add_edge(pydot.Edge(str(source), str(target), label=data["label"]))

    dot_graph.write_raw(str(output_path))
    return graph


def regex_to_dfa(regex: str) -> DeterministicFiniteAutomaton:
    return Regex(regex).to_epsilon_nfa().to_deterministic().minimize()


def graph_to_nfa(
    graph: MultiDiGraph,
    start_states: set[int],
    final_states: set[int],
) -> NondeterministicFiniteAutomaton:
    nodes = set(graph.nodes)
    actual_start_states = start_states or nodes
    actual_final_states = final_states or nodes

    nfa = NondeterministicFiniteAutomaton(
        states=nodes,
        start_state=actual_start_states,
        final_states=actual_final_states,
    )

    for source, target, label in graph.edges(data="label"):
        nfa.add_transition(source, label, target)

    return nfa
