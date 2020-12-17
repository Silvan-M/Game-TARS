from distutils.core import setup
import glob
import py2exe, sys, os

sys.argv.append('py2exe')

setup(
    install_requires=['tensorflow','pygame','numpy'], #external packages as dependencies
    console=["myscript.py"],
    data_files=[("tictactoe", glob.glob("models\\pretrained_model\\*"))
                ("tictactoe", glob.glob("logs\\.keep"))
                ("tictactoe", glob.glob("figures\\.keep"))
                ("snake", glob.glob("models\\pretrained_model\\*"))
                ("snake", glob.glob("logs\\.keep"))
                ("snake", glob.glob("figures\\.keep"))
                ("spaceinvaders", glob.glob("models\\pretrained_model\\*"))
                ("spaceinvaders", glob.glob("logs\\.keep"))
                ("spaceinvaders", glob.glob("figures\\.keep"))
                ("resources", glob.glob("\\*"))
                ],
)