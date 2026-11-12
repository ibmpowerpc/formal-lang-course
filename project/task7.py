from collections import defaultdict

from networkx import DiGraph
from pyformlang.cfg import CFG
from scipy.sparse import csr_matrix, eye

from project.task6 import cfg_to_weak_normal_form


def matrix_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Compute context-free reachability using Boolean matrix multiplication."""
    nodes = tuple(graph.nodes)
    starts = set(nodes) if start_nodes is None else set(start_nodes)
    finals = set(nodes) if final_nodes is None else set(final_nodes)
    if not nodes or not starts or not finals:
        return set()

    grammar = cfg_to_weak_normal_form(cfg)
    node_to_index = {node: index for index, node in enumerate(nodes)}
    size = len(nodes)
    shape = (size, size)
    terminal_rules = defaultdict(set)
    epsilon_heads = set()
    binary_rules = []
    for production in grammar.productions:
        body = production.body
        if not body:
            epsilon_heads.add(production.head)
        elif len(body) == 1:
            terminal_rules[body[0].value].add(production.head)
        else:
            binary_rules.append((production.head, body[0], body[1]))

    coordinates = defaultdict(lambda: ([], []))
    for source, target, label in graph.edges(data="label"):
        for head in terminal_rules.get(label, ()):
            rows, columns = coordinates[head]
            rows.append(node_to_index[source])
            columns.append(node_to_index[target])

    matrices = {
        variable: csr_matrix(shape, dtype=bool)
        for variable in grammar.variables | {grammar.start_symbol}
    }
    for variable, (rows, columns) in coordinates.items():
        matrices[variable] = csr_matrix(
            ([True] * len(rows), (rows, columns)), shape=shape, dtype=bool
        )
    for variable in epsilon_heads:
        matrices[variable] = matrices[variable] + eye(size, dtype=bool, format="csr")

    changed = True
    while changed:
        changed = False
        for head, left, right in binary_rules:
            updated = matrices[head] + matrices[left] @ matrices[right]
            if updated.nnz != matrices[head].nnz:
                matrices[head] = updated.tocsr()
                changed = True

    rows, columns = matrices[grammar.start_symbol].nonzero()
    return {
        (nodes[row], nodes[column])
        for row, column in zip(rows, columns)
        if nodes[row] in starts and nodes[column] in finals
    }
