.. _graphs-introduction:

##############
 Introduction
##############

The `anemoi-graphs` package allows you to design custom graphs for
training data-driven weather models. The graphs are built using a
`recipe`, which is a YAML file that specifies the nodes and edges of the
graph.

*****************
 Data structures
*****************

Several methods are currently supported to create your nodes. You can
use indistinctly any of these to create your `data` or `hidden` nodes.

The `nodes` are defined in the ``nodes`` section of the recipe file. The
keys are the names of the sets of `nodes` that will later be used to
build the connections. Each `nodes` configuration must include a
``node_builder`` section describing how to define the `nodes`. The
following classes define different behaviour:

-  :doc:`node_coordinates/zarr_dataset`
-  :doc:`node_coordinates/npz_file`
-  :doc:`node_coordinates/icon_mesh`
-  :doc:`node_coordinates/tri_refined_icosahedron`
-  :doc:`node_coordinates/hex_refined_icosahedron`
-  :doc:`node_coordinates/healpix`

In addition to the ``node_builder`` section, the `nodes` configuration
can contain an optional ``attributes`` section to define additional node
attributes (weights, mask, ...). For example, the weights can be used to
define the importance of each node in the loss function, or the masks
can be used to build connections only between subsets of nodes.

-  :doc:`node_attributes/weights`
