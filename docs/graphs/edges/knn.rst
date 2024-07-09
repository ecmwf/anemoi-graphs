#####################
 K-Nearest Neighbors
#####################

The knn method is a method to establish connections between two sets of
nodes. Given two set of nodes, (`source`, `target`), the knn method
connects all destination nodes, to its ``num_nearest_neighbours``
closest source nodes.

To use this method to build your connections, you can use the following
YAML configuration:

.. code:: yaml

   edges:
     -  source_name: source
        target_name: destination
        edge_builder:
          _target_: anemoi.graphs.edges.KNNEdges
          num_nearest_neighbours: 3

.. note::

   The knn method is recommended for the decoder edge, to connect all
   data nodes with surrounding hidden nodes.
