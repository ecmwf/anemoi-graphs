.. _overview:

##########
 Overview
##########

A graph :math:`G = (V, E)` is a collection of nodes/vertices :math:`V` and edges :math:`E` that connect the nodes. The nodes can
represent locations in the globe. 

In weather models, the nodes :math:`V` can generally be classified into 2 categories:

- **Data nodes**: The `data nodes` represent the input/output of the data-driven model, so they are linked to existing datasets.
- **Hidden nodes**: These `hidden nodes` represent the latent space, where the internal dynamics are learned.

Similarly, the edges :math:`V` can be classified into 3 categories:

- **Encoder edges**: These `encoder edges` connect the `data` nodes with the `hidden` nodes to encode the input data into the latent space.
- **Processor edges**: These `processor edges` connect the `hidden` nodes with the `hidden` nodes to process the latent space.
- **Decoder edges**: These `decoder edges` connect the `hidden` nodes with the `data` nodes to decode the latent space into the output data.

When building the graph with `anemoi-graphs`, there is no difference between these categories. But it is important to 
have this distinction in mind when designing a weather graph that will be used in a data-driven model with 
:ref:`anemoi-training <anemoi-training:index-page>`.

*******************
 Desing principles
*******************

Particularly, when designing a graph for weather model, you may want to follow the guidelines below:

- Use a coarser resolution for the `hidden nodes`. This will reduce the computational cost of training and inference.
- All input nodes should be connected to the `hidden nodes`. This will ensure that all available information can be used.
- In the encoder edges, minimise the number of connections to the `hidden nodes`. This will reduce the computational cost.
- All output nodes should have incoming connections from a few surrounding `hidden nodes`.
- The number of incoming connections in each set of nodes is desired to be similar, in order to make training more stable.
- Think whether your use case requires long-range connections between the `hidden nodes` or not.

**************
 Installing
**************

To install the package, you can use the following command:

.. code:: bash

   pip install anemoi-graphs[...options...]

The options are:

-  ``dev``: install the development dependencies
-  ``docs``: install the dependencies for the documentation
-  ``test``: install the dependencies for testing
-  ``all``: install all the dependencies

**************
 Contributing
**************

.. code:: bash

   git clone ...
   cd anemoi-graphs
   pip install .[dev]
   pip install -r docs/requirements.txt

You may also have to install pandoc on MacOS:

.. code:: bash

   brew install pandoc
