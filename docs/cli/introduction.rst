.. _cli-introduction:

=================
Command line tool
=================

When you install the `anemoi-graphs` package, a command line tool will also be installed
called ``anemoi-graphs``, which can be used to build graphs based on YAML recipe files,
and inspect existing graphs.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-graphs --help

To **create** a graph, use the ``create`` command:

.. code:: console

   $ anemoi-graphs create recipe.yaml graph.pt

The ``.yaml`` recipe file consists of high-level specifications for generating the graphs at each
layer. An example of a simple recipe file is given in the :ref:`the following section <usage-getting-started>`.

The ``create`` command will read the specifications in the ``recipe.yaml`` recipe file, and write to a PyTorch
``.pt`` file.

To **describe** an existing graph stored as a ``.pt`` file, use the ``describe`` command:

.. code:: console

   $ anemoi-graphs describe graph.pt

This will generate a text summary of the graph, including the number of nodes and edges
at each layer, the geographic boundaries, and statistics about the edge lengths:

.. literalinclude:: ../usage/yaml/global_wo-proc.txt
   :language: console

A set of interactive and static visualisations are generated to allow visual inspection
of the graph design.

Finally, the ``inspect`` command will generate a set of interactive and static visualisations
for visual inspection of the graph design:

.. code:: console

   $ anemoi-graphs inspect graph.pt output_folder/

===================
Command line usage
===================

Create Command
--------------

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: create

Describe Command
----------------
.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: describe

Inspect Command
---------------
.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: inspect
