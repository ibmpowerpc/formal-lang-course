from collections import defaultdict

from networkx import DiGraph
from pyformlang.cfg import CFG
from pyformlang.finite_automaton import EpsilonNFA, Symbol
from pyformlang.rsa import Box, RecursiveAutomaton
from scipy.sparse import csr_matrix, kron


def cfg_to_rsm(cfg: CFG) -> RecursiveAutomaton:
    """Build one component per variable, with a path for each production body."""
    components = {
        variable: EpsilonNFA(start_state={0})
        for variable in cfg.variables | {cfg.start_symbol}
    }
    next_states = defaultdict(lambda: 1)
    for production in cfg.productions:
        automaton = components[production.head]
        state = 0
        for symbol in production.body:
            target = next_states[production.head]
            next_states[production.head] += 1
            automaton.add_transition(state, Symbol(symbol.value), target)
            state = target
        automaton.add_final_state(state)

    return RecursiveAutomaton(
        initial_label=Symbol(cfg.start_symbol.value),
        boxes={
            Box(automaton, Symbol(variable.value))
            for variable, automaton in components.items()
        },
    )


def ebnf_to_rsm(ebnf: str) -> RecursiveAutomaton:
    """Parse regular expressions on production right-hand sides, starting at S."""
    return RecursiveAutomaton.from_text(ebnf)


def _rsm_matrices(rsm: RecursiveAutomaton):
    """Number component states disjointly and decompose transitions by label."""
    coordinates = defaultdict(lambda: ([], []))
    starts = {}
    finals = {}
    size = 0
    for label, box in rsm.boxes.items():
        automaton = box.dfa.to_deterministic()
        indices = {state: size + index for index, state in enumerate(automaton.states)}
        size += len(indices)
        starts[label] = {indices[state] for state in automaton.start_states}
        finals[label] = {indices[state] for state in automaton.final_states}
        for source, transitions in automaton.to_dict().items():
            for symbol, target in transitions.items():
                rows, columns = coordinates[symbol]
                rows.append(indices[source])
                columns.append(indices[target])

    matrices = {
        symbol: csr_matrix(
            ([True] * len(rows), (rows, columns)),
            shape=(size, size),
            dtype=bool,
        )
        for symbol, (rows, columns) in coordinates.items()
    }
    return matrices, starts, finals, size


def _multiple_source_bfs(matrix: csr_matrix, sources: list[int]) -> csr_matrix:
    """Return reachability from each source, including zero-length paths."""
    frontier = csr_matrix(
        ([True] * len(sources), (list(range(len(sources))), sources)),
        shape=(len(sources), matrix.shape[0]),
        dtype=bool,
    )
    visited = frontier.copy()
    while frontier.nnz:
        reached = frontier @ matrix
        frontier = (reached > visited).tocsr()
        visited = visited + frontier
    return visited


def tensor_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Saturate graph relations using tensor products and MSBFS, as in lecture 16."""
    nodes = tuple(graph.nodes)
    starts = set(nodes) if start_nodes is None else set(start_nodes)
    finals = set(nodes) if final_nodes is None else set(final_nodes)
    if not nodes or not starts or not finals:
        return set()

    node_indices = {node: index for index, node in enumerate(nodes)}
    size = len(nodes)
    shape = (size, size)
    rsm_matrices, box_starts, box_finals, rsm_size = _rsm_matrices(rsm)
    if not rsm_size:
        return set()

    coordinates = defaultdict(lambda: ([], []))
    for source, target, label in graph.edges(data="label"):
        rows, columns = coordinates[Symbol(label)]
        rows.append(node_indices[source])
        columns.append(node_indices[target])
    graph_matrices = {
        symbol: csr_matrix(
            ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
        )
        for symbol, (rows, columns) in coordinates.items()
    }
    for label in rsm.labels:
        graph_matrices.setdefault(label, csr_matrix(shape, dtype=bool))

    # Product states are (vertex, RSM state), numbered vertex * rsm_size + state.
    source_info = [
        (vertex, label, state)
        for vertex in range(size)
        for label, states in box_starts.items()
        for state in states
    ]
    sources = [vertex * rsm_size + state for vertex, _, state in source_info]
    product_size = size * rsm_size

    while True:
        product = csr_matrix((product_size, product_size), dtype=bool)
        for symbol in graph_matrices.keys() & rsm_matrices.keys():
            product = product + kron(
                graph_matrices[symbol], rsm_matrices[symbol], format="csr"
            )

        reachable = _multiple_source_bfs(product, sources)
        coordinates = defaultdict(lambda: ([], []))
        rows, columns = reachable.nonzero()
        for row, column in zip(rows, columns):
            source, label, _ = source_info[row]
            target, state = divmod(column, rsm_size)
            if state in box_finals[label]:
                starts_found, ends_found = coordinates[label]
                starts_found.append(source)
                ends_found.append(target)

        changed = False
        for label, (rows, columns) in coordinates.items():
            found = csr_matrix(
                ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
            )
            updated = graph_matrices[label] + found
            if updated.nnz != graph_matrices[label].nnz:
                graph_matrices[label] = updated.tocsr()
                changed = True
        if not changed:
            break

    rows, columns = graph_matrices[rsm.initial_label].nonzero()
    return {
        (nodes[row], nodes[column])
        for row, column in zip(rows, columns)
        if nodes[row] in starts and nodes[column] in finals
    }
