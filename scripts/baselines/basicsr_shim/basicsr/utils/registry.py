class Registry:
    def __init__(self, name):
        self.name, self._obj_map = name, {}

    def register(self, obj=None):
        if obj is None:
            return lambda o: self._obj_map.setdefault(o.__name__, o)
        self._obj_map[obj.__name__] = obj
        return obj

    def get(self, name):
        return self._obj_map[name]


ARCH_REGISTRY = Registry("arch")
