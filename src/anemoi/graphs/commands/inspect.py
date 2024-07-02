from anemoi.graphs.inspector import GraphInspectorTool

from . import Command


class Inspect(Command):
    """Inspect a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument("graph", help="Path to the graph (a .PT file).")
        command_parser.add_argument("output_path", help="Path to store the inspection results.")

    def run(self, args):
        kwargs = vars(args)

        inspector = GraphInspectorTool(**kwargs)
        inspector.run_all()


command = Inspect
