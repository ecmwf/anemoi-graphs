################################
 Triangular refined Icosahedron
################################

This class allows us to define nodes based on iterative refinements of
an icoshaedron with triangles.

To define the `node coordinates` based on icosahedral refinements of an
icosahedron, you can use the following YAML configuration:

.. code:: yaml

   nodes:
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.TriRefinedIcosahedronNodes
         resolution: 4
       attributes: ...

where resolution is the number of refinements to be applied to the
icosahedron.

.. warning::

   This class will require the `trimesh <https://trimesh.org>`_ package
   to be installed. You can install it with `pip install trimesh`.
