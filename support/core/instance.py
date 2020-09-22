"""
"""

__all__ = [
    "Instance"
]

from .utils import pretty_table_printer


class Instance(object):
    """
    """

    def __init__(self, **fields):

        self.fields = fields

    def add_field(self, field_name, field):
        """
        """
        self.fields[field_name] = field

    def items(self):
        """
        """
        return self.fields.items()

    def __contains__(self, item):
        return item in self.fields

    def __getitem__(self, name):
        if name in self.fields:
            return self.fields[name]
        else:
            raise KeyError("{} not found".format(name))

    def __setitem__(self, name, field):
        return self.add_field(name, field)

    def __repr__(self):
        return str(pretty_table_printer(self))
