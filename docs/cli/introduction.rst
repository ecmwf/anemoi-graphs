.. _cli-introduction:

=================
Command line tool
=================

When you install the `anemoi-graphs` package, a command line tool will also be installed
called ``anemoi-graphs``, which can be used to design and inspect weather graphs.

The tool can provide help with the ``--help`` options:

.. code-block:: bash

    % anemoi-graphs --help

The available commands are as follows:

* ``create`` creates a graph from a recipe file.
* ``describe`` describes a graph stored in your filesystem. It will print graph information to the console.
* ``inspect`` inspects a graph stored in your filesystem. A set of interactive and static 
  visualisations are generated to allow visual inspection of the graph design.

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :nosubcommands:

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: create

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: describe

.. argparse::
    :module: anemoi.graphs.__main__
    :func: create_parser
    :prog: anemoi-graphs
    :path: inspect