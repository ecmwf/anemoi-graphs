.. _node-attributes:

####################
 Nodes - Attributes
####################

.. warning::

   This is still a work in progress. More classes will be added in the
   future.

The nodes :math:`V` correspond to locations on the earth's surface.
Apart from defining its locations, the `nodes` can contain additional
attributes, that should be defined in the ``attributes`` section of the
nodes configuration. For example, a `weights` attribute can be used to
define the importance of each node in the loss function, or a `masks`
attribute can be used to build connections only between subsets of
nodes.

.. toctree::
   :maxdepth: 1

   node_attributes/weights
