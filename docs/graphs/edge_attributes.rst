.. _edge-attributes:

####################
 Edges - Attributes
####################

There are 2 main edge attributes implemented in the `anemoi-graphs` package:



**************
 Edge length
**************

The `edge length` is a scalar value that represents the distance between the source and target nodes. This attribute is
calculated using the Haversine formula, which is a method to calculate the distance between two points on the Earth 
surface given their latitude and longitude coordinates.

.. code:: yaml

   edges:
     - nodes: ...
       edge_builder: ...
       attributes:
         edge_length:
            _target_: anemoi.graphs.edges.attributes.EdgeLength


****************
 Edge direction
****************

The `edge direction` is a 2D vector that represents the direction of the edge. This attribute is calculated using the
difference between the latitude and longitude coordinates of the source and target nodes.

.. code:: yaml

   edges:
     - nodes: ...
       edge_builder: ...
       attributes:
         edge_length:
            _target_: anemoi.graphs.edges.attributes.EdgeDirection
