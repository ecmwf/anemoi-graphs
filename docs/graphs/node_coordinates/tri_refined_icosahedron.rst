################################
 Triangular refined Icosahedron
################################

To define the `node coordinates` based on icosahedral refinements of an icosahedron, you can use the following YAML 
configuration:

.. code-block:: yaml

    nodes:
      data:
        coords:
          _target_: anemoi.graphs.nodes.TriRefinedIcosahedronNodeBuilder
          resolution: 4

where resolution is the number of refinements to apply to the icosahedron.

The class ``TriRefinedIcosahedronNodeBuilder`` requires `trimesh <https://trimesh.org>`_ __Python__ package.