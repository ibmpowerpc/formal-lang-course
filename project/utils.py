from collections.abc import Iterable
from pathlib import Path

import cfpq_data
import pydot
from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
    State,
    Symbol,
)
from pyformlang.regular_expression import Regex
from scipy.sparse import csr_matrix, eye, kron


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


class AdjacencyMatrixFA:
    def __init__(self, automaton: NondeterministicFiniteAutomaton):
        self.states = tuple(automaton.states)
        self.state_to_index = {
            state: index for index, state in enumerate(self.states)
        }
        self.start_states = {
            self.state_to_index[state] for state in automaton.start_states
        }
        self.final_states = {
            self.state_to_index[state] for state in automaton.final_states
        }
        self.matrices = self._build_matrices(automaton)

    def _build_matrices(
        self, automaton: NondeterministicFiniteAutomaton
    ) -> dict[Symbol, csr_matrix]:
        transitions: dict[Symbol, tuple[list[int], list[int]]] = {}

        for source, transitions_by_symbol in automaton.to_dict().items():
            for symbol, targets in transitions_by_symbol.items():
                if isinstance(targets, State):
                    targets = {targets}

                rows, columns = transitions.setdefault(symbol, ([], []))
                for target in targets:
                    rows.append(self.state_to_index[source])
                    columns.append(self.state_to_index[target])

        size = len(self.states)
        return {
            symbol: csr_matrix(
                ([True] * len(rows), (rows, columns)),
                shape=(size, size),
                dtype=bool,
            )
            for symbol, (rows, columns) in transitions.items()
        }

    @classmethod
    def _from_components(
        cls,
        states: tuple[object, ...],
        start_states: set[int],
        final_states: set[int],
        matrices: dict[Symbol, csr_matrix],
    ) -> "AdjacencyMatrixFA":
        automaton = cls.__new__(cls)
        automaton.states = states
        automaton.state_to_index = {
            state: index for index, state in enumerate(states)
        }
        automaton.start_states = start_states
        automaton.final_states = final_states
        automaton.matrices = matrices
        return automaton

    def accepts(self, word: Iterable[Symbol]) -> bool:
        size = len(self.states)
        if size == 0 or not self.start_states:
            return False

        start_states = list(self.start_states)
        current_states = csr_matrix(
            (
                [True] * len(start_states),
                ([0] * len(start_states), start_states),
            ),
            shape=(1, size),
            dtype=bool,
        )

        for symbol in word:
            matrix = self.matrices.get(Symbol(symbol))
            if matrix is None:
                return False
            current_states = current_states @ matrix
            if current_states.nnz == 0:
                return False

        return any(index in self.final_states for index in current_states.indices)

    def is_empty(self) -> bool:
        if not self.start_states or not self.final_states:
            return True

        closure = self._transitive_closure()
        for start_state in self.start_states:
            reachable = set(closure.getrow(start_state).indices)
            if reachable & self.final_states:
                return False
        return True

    def _transitive_closure(self) -> csr_matrix:
        size = len(self.states)
        closure = eye(size, dtype=bool, format="csr")

        for matrix in self.matrices.values():
            closure = (closure + matrix).astype(bool)

        while True:
            previous_nonzero_count = closure.nnz
            closure = (closure + closure @ closure).astype(bool).tocsr()
            if closure.nnz == previous_nonzero_count:
                return closure


def intersect_automata(
    automaton1: AdjacencyMatrixFA,
    automaton2: AdjacencyMatrixFA,
) -> AdjacencyMatrixFA:
    second_size = len(automaton2.states)
    states = tuple(
        (first_state, second_state)
        for first_state in automaton1.states
        for second_state in automaton2.states
    )
    start_states = {
        first * second_size + second
        for first in automaton1.start_states
        for second in automaton2.start_states
    }
    final_states = {
        first * second_size + second
        for first in automaton1.final_states
        for second in automaton2.final_states
    }
    common_symbols = automaton1.matrices.keys() & automaton2.matrices.keys()
    matrices = {
        symbol: kron(
            automaton1.matrices[symbol],
            automaton2.matrices[symbol],
            format="csr",
        )
        for symbol in common_symbols
    }

    return AdjacencyMatrixFA._from_components(
        states,
        start_states,
        final_states,
        matrices,
    )


def tensor_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_automaton = AdjacencyMatrixFA(
        graph_to_nfa(graph, start_nodes, final_nodes)
    )
    regex_automaton = AdjacencyMatrixFA(regex_to_dfa(regex))
    intersection = intersect_automata(graph_automaton, regex_automaton)
    closure = intersection._transitive_closure()

    result = set()
    for start_state in intersection.start_states:
        graph_start = intersection.states[start_state][0].value
        reachable_final_states = (
            set(closure.getrow(start_state).indices) & intersection.final_states
        )
        for final_state in reachable_final_states:
            graph_final = intersection.states[final_state][0].value
            result.add((graph_start, graph_final))

    return result
