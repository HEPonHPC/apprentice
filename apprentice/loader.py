def loadPlugin(model, *args, **kwargs):
    r"""
    Plugin loader. This attemts to load and return and instance of
    the class CLASS in the file FILENAME. Both are given as a single
    string using : as separator. The args and kwargs are passed on to
    the base class.

        :Arguments:
            * *model* (``str``) --
              FILENAME:CLASS

    """
    plugin_file, plugin_name = model.split(":")
    import os
    if not os.path.exists(plugin_file):
        raise Exception("Specified (module) file '{}' does not exist.{}".format(plugin_file))

    # https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path
    from importlib.machinery import SourceFileLoader

    foo = SourceFileLoader(plugin_name, plugin_file).load_module()

    # https://stackoverflow.com/questions/1796180/how-can-i-get-a-list-of-all-classes-within-current-module-in-python
    import inspect

    IM = dict(inspect.getmembers(foo, inspect.isclass))
    if not plugin_name in IM:
        raise Exception("Specified class '{}'  not found in module {}".format(plugin_name, plugin_file))

    return IM[plugin_name](*args, **kwargs)
