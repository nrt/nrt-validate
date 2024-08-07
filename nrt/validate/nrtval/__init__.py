"""
nrt-validate commandline interface components
"""
import click
import os
from nrt.validate import DemoNotebook

@click.group()
def cli():
    pass

@cli.command()
def demo():
    """Run the demo notebook using Voilà"""
    notebook_path = DemoNotebook.path()
    os.system(f'voila {notebook_path} --theme=dark')

if __name__ == "__main__":
    cli()

