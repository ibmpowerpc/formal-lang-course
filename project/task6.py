from collections import defaultdict, deque

from networkx import DiGraph
from pyformlang.cfg import CFG, Production, Variable

from project.task2 import graph_to_nfa


def cfg_to_weak_normal_form(cfg: CFG) -> CFG:
    """Return a Chomsky normal form preserving the empty word."""
    normal_form = cfg.to_normal_form()
    productions = set(normal_form.productions)
    if cfg.generate_epsilon():
        productions.add(Production(cfg.start_symbol, []))
    return CFG(start_symbol=cfg.start_symbol, productions=productions)


def hellings_based_cfpq(
    cfg: CFG,
    graph: DiGraph,
    start_nodes: set[int] | None = None,
    final_nodes: set[int] | None = None,
) -> set[tuple[int, int]]:
    starts = set(graph.nodes) if start_nodes is None else set(start_nodes)
    finals = set(graph.nodes) if final_nodes is None else set(final_nodes)
    if not starts or not finals:
        return set()

    grammar = cfg_to_weak_normal_form(cfg)
    terminal_rules = defaultdict(set)
    left_rules = defaultdict(set)
    right_rules = defaultdict(set)
    epsilon_heads = set()
    for production in grammar.productions:
        body = production.body
        if not body:
            epsilon_heads.add(production.head)
        elif len(body) == 1:
            terminal_rules[body[0].value].add(production.head)
        else:
            left, right = body
            left_rules[left].add((production.head, right))
            right_rules[right].add((production.head, left))

    outgoing: dict[tuple[int, Variable], set[int]] = defaultdict(set)
    incoming: dict[tuple[int, Variable], set[int]] = defaultdict(set)
    queue = deque()

    def add(variable: Variable, source: int, target: int):
        if target not in outgoing[source, variable]:
            outgoing[source, variable].add(target)
            incoming[target, variable].add(source)
            queue.append((variable, source, target))

    for variable in epsilon_heads:
        for node in graph.nodes:
            add(variable, node, node)

    graph_automaton = graph_to_nfa(graph, set(), set())
    for source, transitions in graph_automaton.to_dict().items():
        for symbol, targets in transitions.items():
            for variable in terminal_rules.get(symbol.value, ()):
                for target in targets:
                    add(variable, source.value, target.value)

    while queue:
        variable, source, target = queue.popleft()
        for head, right in left_rules.get(variable, ()):
            for end in tuple(outgoing.get((target, right), ())):
                add(head, source, end)
        for head, left in right_rules.get(variable, ()):
            for begin in tuple(incoming.get((source, left), ())):
                add(head, begin, target)

    return {
        (source, target)
        for source in starts
        for target in outgoing.get((source, grammar.start_symbol), ())
        if target in finals
    }
