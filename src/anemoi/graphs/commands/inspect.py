from anemoi.graphs.inspector import GraphDescription
from anemoi.graphs.inspector import GraphInspectorTool

from . import Command


class Inspect(Command):
    """Inspect a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--show_attribute_distributions",
            action="store_false",
            help="Show distribution of edge/node attributes.",
        )
        command_parser.add_argument("--description", action="store_false", help="Show the description of the graph.")
        command_parser.add_argument("path", help="Path to the graph (a .PT file).")
        command_parser.add_argument("output_path", help="Path to store the inspection results.")

    def run(self, args):
        kwargs = vars(args)

        if kwargs.get("description", False):
            GraphDescription(kwargs["path"]).describe()

        inspector = GraphInspectorTool(**kwargs)
        inspector.run_all()


command = Inspect
