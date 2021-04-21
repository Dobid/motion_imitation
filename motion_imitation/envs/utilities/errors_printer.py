import pathlib
import inspect

def printDebug(message: str):
    frame = inspect.stack()[1]
    module = inspect.getmodule(frame[0])
    filename = module.__file__
    print(str + " file : ", )