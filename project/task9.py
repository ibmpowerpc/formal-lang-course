"""GLL-based context-free path querying.

The implementation follows the GLL scheme from the lecture: descriptors are
triples ``(graph vertex, RSM state, GSS node)``, GSS edges keep the return
state, and saved pops replay already finished calls when a new return edge
appears.  Only reachability is computed, so the SPPF/matched range machinery
is not needed.
"""

from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.finite_automaton import Symbol
from pyformlang.rsa import RecursiveAutomaton


def _box_view(box) -> tuple:
    """Return the start state, final states and transitions of a box."""
    automaton = box.dfa.to_deterministic()
    transitions = defaultdict(dict)
    for source, outgoing in automaton.to_dict().items():
        for symbol, target in outgoing.items():
            transitions[source][Symbol(symbol)] = target
    return automaton.start_state, set(automaton.final_states), transitions


def gll_based_cfpq(
    rsm: RecursiveAutomaton,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    """Find reachable pairs with GLL, as described in the lecture."""
    nodes = set(graph.nodes)
    starts = set(nodes) if start_nodes is None else set(start_nodes)
    finals = set(nodes) if final_nodes is None else set(final_nodes)
    if not starts or not finals:
        return set()

    boxes = {label: _box_view(box) for label, box in rsm.boxes.items()}
    initial_label = rsm.initial_label
    if initial_label not in boxes:
        return set()

    # Graph adjacency grouped by the edge label.
    adjacency = defaultdict(lambda: defaultdict(list))
    for source, target, data in graph.edges(data=True):
        label = data.get("label")
        if label is not None:
            adjacency[source][Symbol(label)].append(target)

    # A GSS node is a call context: graph vertex, called label and start state.
    label_of = {}

    def node_for(vertex, label, state):
        node = (vertex, label, state)
        if node not in label_of:
            label_of[node] = label
        return node

    incoming = defaultdict(list)
    edges_seen = set()
    popped = defaultdict(set)
    roots = {}
    root_sources = {}

    descriptors = deque()
    seen = set()

    def add_descriptor(vertex, state, node):
        descriptor = (vertex, state, node)
        if descriptor not in seen:
            seen.add(descriptor)
            descriptors.append(descriptor)

    def add_edge(caller, label, return_state, callee):
        edge = (caller, label, return_state, callee)
        if edge in edges_seen:
            return
        edges_seen.add(edge)
        incoming[callee].append((label, return_state, caller))
        # Saved pops: replay calls that finished before this edge appeared.
        for vertex, _ in popped[callee]:
            add_descriptor(vertex, return_state, caller)

    for source in starts:
        start_state = boxes[initial_label][0]
        root = node_for(source, initial_label, start_state)
        roots[source] = root
        root_sources[root] = source
        add_descriptor(source, start_state, root)

    result = set()
    while descriptors:
        vertex, state, node = descriptors.popleft()
        label = label_of[node]
        _, box_finals, transitions = boxes[label]

        if state in box_finals:
            popped[node].add((vertex, state))
            source = root_sources.get(node)
            if source is not None and vertex in finals:
                result.add((source, vertex))
            for edge_label, return_state, caller in incoming[node]:
                if edge_label == label:
                    add_descriptor(vertex, return_state, caller)

        for symbol, target in transitions.get(state, {}).items():
            if symbol in boxes:
                callee = node_for(vertex, symbol, boxes[symbol][0])
                add_edge(node, symbol, target, callee)
                add_descriptor(vertex, boxes[symbol][0], callee)
            else:
                for neighbor in adjacency[vertex].get(symbol, ()):
                    add_descriptor(neighbor, target, node)

    return result
