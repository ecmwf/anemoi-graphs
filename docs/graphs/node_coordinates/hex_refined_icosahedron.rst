###############################
 Hexagonal refined Icosahedron
###############################

To define the `node coordinates` based on hexagonal refinements of an
icosahedron, you can use the following YAML configuration:

.. code:: yaml

   nodes:
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.HexRefinedIcosahedronNodes
         resolution: 4

where resolution is the number of refinements to apply.

The class ``HexRefinedIcosahedronNodes`` requires `h3 <https://h3.org>`_
__Python__ package.
