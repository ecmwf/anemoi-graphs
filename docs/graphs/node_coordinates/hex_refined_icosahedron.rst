################################
 Hexagonal refined Icosahedron
################################

To define the `node coordinates` based on hexagonal refinements of an icosahedron, you can use the following YAML 
configuration:

.. code-block:: yaml

    nodes:
      data:
        coords:
          _target_: anemoi.graphs.nodes.HexRefinedIcosahedronNodeBuilder
          resolution: 4

where resolution is the number of refinements to apply.

The class ``HexRefinedIcosahedronNodeBuilder`` requires `h3 <https://h3.org>`_ __Python__ package.