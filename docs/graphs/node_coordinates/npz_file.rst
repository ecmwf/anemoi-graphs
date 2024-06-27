###############
 From NPZ file
###############

To define the `node coordinates` based on a NPZ file, you can use the following YAML configuration:

.. code-block:: yaml

    nodes:
      data:
        coords:
          _target_: anemoi.graphs.nodes.NPZFileNodeBuilder
          grids_definition_path: path/to/folder/with/grids/
          resolution: o48

where `grids_definition_path` is the path to the folder containing the grids definition files and `resolution` is
the resolution of the grid to use. 

By default, the grids file are supposed to be in the `grids` folder in the same directory as the recipe file. The grids 
definition files are expected to be name `"grid_{resolution}.npz"`.