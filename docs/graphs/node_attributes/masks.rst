#######
 Masks
#######

`Masks` are a special type of node attribute. They are useful to filter
the connections between set of nodes.

At the moment, the mask's attribute can only be generated for nodes
defined from ``ZarrDatasetNodes``.

*********************
 From missing values
*********************

To generate a `mask` from missing values, you need to include the
`MissingMask` class into the ``attributes`` section in recipe file.

.. literalinclude:: ../yaml/attributes_bool-masks.yaml
   :language: yaml

.. warning::

   The missing values should be fixed over time. If not, the mask will
   be generated only from the first time step.

***********************
 From boolean variable
***********************

To use a boolean variable as a mask, you need to include the
`BooleanMask` class into the ``attributes`` section in the recipe file.

.. literalinclude:: ../yaml/attributes_missing-masks.yaml
   :language: yaml

.. warning::

   The boolean variable should be static over time. If not, the mask
   will be generated only from the first time step.

***********************
 From cutout operation
***********************

To generate a `mask` from a cutout operation, you need to include the
`CutoutMask` class into the ``attributes`` section in the recipe file.

.. literalinclude:: ../yaml/attributes_cutout-masks.yaml
   :language: yaml

.. warning::

   The cutout operation should be done in the corresponding nodes to
   build this attribute.
