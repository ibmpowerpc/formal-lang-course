from networkx import MultiDiGraph
from pyformlang.finite_automaton import (
    DeterministicFiniteAutomaton,
    NondeterministicFiniteAutomaton,
)
from pyformlang.regular_expression import Regex


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
