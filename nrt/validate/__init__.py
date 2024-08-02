import os


__version__ = "0.0.1"


class DemoNotebook:
    @staticmethod
    def path():
        return os.path.join(os.path.dirname(__file__), 'notebooks', 'demo_notebook.ipynb')

