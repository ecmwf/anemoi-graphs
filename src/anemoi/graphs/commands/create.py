from anemoi.graphs.create import GraphCreator

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
        command_parser.add_argument("config", help="Configuration yaml file defining the recipe to create the graph.")
        command_parser.add_argument("path", help="Path to store the created graph.")

    def run(self, args):
        kwargs = vars(args)

        c = GraphCreator(**kwargs)
        c.create()


command = Create
