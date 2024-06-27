.. _graphs-introduction:

##############
 Introduction
##############

The `anemoi-graphs` package allows you to design custom graphs
for training data-driven weather models. The graphs are built using
a `recipe`, which is a YAML file that specifies the nodes and edges 
of the graph. 


**********
 Concepts
**********

nodes
    A `node` represents a location (2D) in the Earth surface which may contain
    additional `attributes`.

data nodes
    A set of nodes that represents one or multiple datasets. The `data nodes` may
    correspond to the input/output of our data-driven model. They can be define from 
    Zarr datasets and it is compatible with :ref:`anemoi-datasets <anemoi-datasets:index-page>` 
    operations such as `cutout` or `thinning`.

hidden nodes
    The `hidden nodes` capture intermediate representations of the model, which are used
    to learn the dynamics of the system considered (athmosphere, ocean, etc, ...). These
    nodes can be generated from existing locations (Zarr datasets or NPZ files) or 
    algorithmically from iterative refinements of polygons over the globe. 

orphan nodes
    A set of nodes that are not connected to any other node in the graph. These nodes
    can be used to store additional information that is not directly used in the training
    process.

edges
    An `edge` represents a connection between two nodes. The `edges` can be used to
    define the flow of information between the nodes. Edges may also contains `attributes`
    related to its length, direction or other properties.



*****************
 Data structures
*****************

The nodes :math:`V` correspond to locations in the Earth surface, and they can be classified into 2 categories:

- **Data nodes**: The `data nodes` represent the input/output of the data-driven model, so they are linked to existing datasets.
- **Hidden nodes**: These `hidden nodes` represent the latent space, where the internal dynamics are learned.

Currently, there are several methods supported to create your nodes. You can use indistinctly any of these to create
your `data` or `hidden` nodes.

The `nodes` are defined in the ``nodes`` section of the recipe file. The keys will be the names of the sets of `nodes`,
that will later be used to build the connections. Each nodes configuration must include a ``coords`` section describing
how to generate the `nodes`. The following classes define different behaviours:

- :doc:`node_coordinates/zarr_dataset`
- :doc:`node_coordinates/npz_file`
- :doc:`node_coordinates/tri_refined_icosahedron`
- :doc:`node_coordinates/hex_refined_icosahedron`
- :doc:`node_coordinates/stretched_tri_refined_icosahedron`
- :doc:`node_coordinates/rectilinear`

Apart from the ``coords`` section, the nodes configuration may contain an optional ``attributes`` section to define
additional node attributes (weights, mask, ...). For example, the weights can be used to define the importance of each 
node in the loss function, or the masks can be used to build connections only between subsets of nodes.

- :doc:`node_attributes/masks`
- :doc:`node_attributes/weights`
