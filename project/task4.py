from networkx import MultiDiGraph
from scipy.sparse import csr_matrix

from project.task2 import graph_to_nfa, regex_to_dfa
from project.task3 import AdjacencyMatrixFA


def ms_bfs_based_rpq(
    regex: str,
    graph: MultiDiGraph,
    start_nodes: set[int],
    final_nodes: set[int],
) -> set[tuple[int, int]]:
    graph_automaton = AdjacencyMatrixFA(graph_to_nfa(graph, start_nodes, final_nodes))
    dfa = regex_to_dfa(regex)
    sources = tuple(graph_automaton.start_states)
    if not sources or not graph_automaton.final_states:
        return set()

    shape = (len(sources), len(graph_automaton.states))
    initial_frontier = csr_matrix(
        ([True] * len(sources), (list(range(len(sources))), sources)),
        shape=shape,
        dtype=bool,
    )
    visited = {state: csr_matrix(shape, dtype=bool) for state in dfa.states}
    frontier = {state: initial_frontier.copy() for state in dfa.start_states}
    for state, matrix in frontier.items():
        visited[state] = matrix.copy()

    transitions = dfa.to_dict()
    while frontier:
        candidates = {}
        for state, matrix in frontier.items():
            for symbol, target in transitions.get(state, {}).items():
                graph_matrix = graph_automaton.matrices.get(symbol)
                if graph_matrix is None:
                    continue
                reached = matrix @ graph_matrix
                if target in candidates:
                    candidates[target] = candidates[target] + reached
                else:
                    candidates[target] = reached

        frontier = {}
        for state, matrix in candidates.items():
            unseen = (matrix > visited[state]).tocsr()
            if unseen.nnz:
                frontier[state] = unseen
                visited[state] = visited[state] + unseen

    result = set()
    for state in dfa.final_states:
        rows, columns = visited[state].nonzero()
        for row, column in zip(rows, columns):
            if column in graph_automaton.final_states:
                source = graph_automaton.states[sources[row]].value
                target = graph_automaton.states[column].value
                result.add((source, target))
    return result
