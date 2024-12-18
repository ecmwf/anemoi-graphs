.. _anemoi-graphs:

.. _index-page:

###############################################
 Welcome to the `anemoi-graphs` documentation!
###############################################

.. warning::

   This documentation is work in progress.

The *anemoi-graphs* package is a collection of tools enabling you to
design custom graphs for training data-driven weather models. It is one
of the packages within the `anemoi framework
<https://anemoi-docs.readthedocs.io/en/latest/>`_.

**************
 About Anemoi
**************

*Anemoi* is a framework for developing machine learning weather
forecasting models. It comprises of components or packages for preparing
training datasets, conducting ML model training and a registry for
datasets and trained models. *Anemoi* provides tools for operational
inference, including interfacing to verification software. As a
framework it seeks to handle many of the complexities that
meteorological organisations will share, allowing them to easily train
models from existing recipes but with their own data.

****************
 Quick overview
****************

The *anemoi-graphs* package contains a suite of tools for creating
graphs for use in data-driven weather forecasting models, typically
those based on deep learning approaches.

*anemoi-graphs* provides a simple high-level interface based on a YAML
recipe file, which can be used to build graphs for the input, hidden and
output layers. For each layer, the package allows you to:

-  :ref:`Define graph nodes <graphs-node_coordinates>` based on
   coordinates defined in a dataset (Zarr and NPZ) or via algorithmic
   approaches such as the triangular refined icosahedron.

-  :ref:`Define edges <graphs-edges>` (connections between nodes) based
   on methods such as the cut-off radius or K nearest-neighbours.

-  :ref:`Define attributes <graphs-node_attributes>` of nodes and edges,
   such as weights, lengths and directions.

The node definition also allows to combine two input datasets, enabling
limited-area models and stretched grid models.

The specification of each layer is defined using a YAML file, which is
run via the :ref:`command-line tool <cli-introduction>`. The
command-line tool allows you to quickly create graphs, as well as
describe and inspect existing graphs, from command line prompts. For
example, to create a graph based on an existing YAML file
``recipe.yaml`` and output to ``graph.pt``:

.. code:: console

   $ anemoi-graphs create recipe.yaml graph.pt

In the rest of this documentation, the commands and syntax for defining
the YAML files will be explained. A full example of a YAML file for a
global weather model is found in the :ref:`usage-getting-started`
section.

************
 Installing
************

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

***********************
 Other Anemoi packages
***********************

-  :ref:`anemoi-utils <anemoi-utils:index-page>`
-  :ref:`anemoi-transform <anemoi-transform:index-page>`
-  :ref:`anemoi-datasets <anemoi-datasets:index-page>`
-  :ref:`anemoi-models <anemoi-models:index-page>`
-  :ref:`anemoi-training <anemoi-training:index-page>`
-  :ref:`anemoi-inference <anemoi-inference:index-page>`
-  :ref:`anemoi-registry <anemoi-registry:index-page>`

*********
 License
*********

*Anemoi* is available under the open source `Apache License`__.

.. __: http://www.apache.org/licenses/LICENSE-2.0.html

..
   ..................................................................................

..
   From here defines the TOC in the sidebar, but is not rendered directly on the page.

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Introduction

   overview
   cli/introduction

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Building graphs

   graphs/introduction
   graphs/node_coordinates
   graphs/node_attributes
   graphs/edges
   graphs/edge_attributes
   graphs/post_processor

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Recipe examples

   usage/getting_started
   usage/limited_area

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Modules

   modules/node_builder
   modules/edge_builder
   modules/node_attributes
   modules/edge_attributes
   modules/graph_creator
   modules/graph_inspector
   modules/post_processor

.. toctree::
   :maxdepth: 1
   :hidden:
   :caption: Developing Anemoi Graphs

   dev/contributing
   dev/code_structure
   dev/testing
