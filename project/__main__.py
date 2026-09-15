import cfpq_data


def count_by_name(name: str):
    path = cfpq_data.download(name)
    graph = cfpq_data.graph_from_csv(path)
    return (
        graph.number_of_nodes(),
        graph.number_of_edges(),
        {data["label"] for _, _, data in graph.edges(data=True)},
    )


def build_two_cycled_graph(n: int, m: int, labels):
    return cfpq_data.labeled_two_cycles_graph(n, m, labels=labels)
