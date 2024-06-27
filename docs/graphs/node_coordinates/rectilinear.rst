#########################
 Rectilinear grid
#########################

.. warning::
    This is only expected to be used in some Limited Area Models (LAMs) and is not recommended for global graphs.

To define the `node coordinates` based on rectangular refinements, you can use the following YAML configuration:

.. code-block:: yaml

    nodes:
      data:
        coords:
          _target_: anemoi.graphs.nodes.RectilinearNodeBuilder

where resolution is the number of refinements to apply to the rectangular grid.