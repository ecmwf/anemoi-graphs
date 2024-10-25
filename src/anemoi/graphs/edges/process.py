import torch


def concat_edges(edge_index1: torch.Tensor, edge_index2: torch.Tensor) -> torch.Tensor:
    """Concat edges

    Parameters
    ----------
    edge_index1: torch.Tensor
        Edge indices of the first set of edges. Shape: (2, num_edges1)
    edge_index2: torch.Tensor
        Edge indices of the second set of edges. Shape: (2, num_edges2)

    Returns
    -------
    torch.Tensor
        Concatenated edge indices.
    """
    edge_index = torch.cat([edge_index1, edge_index2], axis=1)

    # Remove repeated nodes
    unique_edges = set(map(tuple, edge_index.t().tolist()))
    edge_index = torch.tensor(list(unique_edges), dtype=torch.long).t()

    return edge_index
