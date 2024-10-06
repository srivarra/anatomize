import datatree as dt


@dt.register_datatree_accessor("an")
class AnatomizeAccessor:
    """DataTree Accessor for anatomize."""

    def __init__(self, datatree_obj: dt.DataTree):
        self._obj = datatree_obj
