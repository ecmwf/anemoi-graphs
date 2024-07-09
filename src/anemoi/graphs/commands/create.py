from anemoi.graphs.create import GraphCreator
from anemoi.graphs.create import GraphDescription

from . import Command


class Create(Command):
    """Create a graph."""

    internal = True
    timestamp = True

    def add_arguments(self, command_parser):
        command_parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite existing files. This will delete the target graph if it already exists.",
        )
        command_parser.add_argument("--description", action="store_false", help="Show the description of the graph.")
        command_parser.add_argument("config", help="Configuration yaml file defining the recipe to create the graph.")
        command_parser.add_argument("path", help="Path to store the created graph.")

    def run(self, args):
        kwargs = vars(args)

        c = GraphCreator(**kwargs)
        c.create()

        if kwargs.get("description", False):
            GraphDescription(kwargs["path"]).describe()


command = Create
