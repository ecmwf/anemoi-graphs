##########################################
 Stretched triangular refined Icosahedron
##########################################

To define the `node coordinates` based on stretched icosahedral
refinements, you can use the following YAML configuration:

.. code:: yaml

   nodes:
     data:
       node_builder:
         _target_: anemoi.graphs.nodes.StretchedTriRefinedIcosahedralNodes
         resolution: [4, 8]

where resolution is the number of refinements to apply to the
icosahedron on each area. The first resolution corresponds to the base
resolution, and the second resolution corresponds to the resolution at
the area of interest.

The class ``StretchedTriRefinedIcosahedronNodes`` requires `trimesh
<https://trimesh.org>`_ __Python__ package.
