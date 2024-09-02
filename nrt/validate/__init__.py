import os


__path__ = __import__('pkgutil').extend_path(__path__, __name__)

class DemoNotebook:
    @staticmethod
    def path():
        return os.path.join(os.path.dirname(__file__), 'notebooks', 'demo_notebook.ipynb')

