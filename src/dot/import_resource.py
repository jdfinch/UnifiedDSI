
import types


def to_module(code: str, name: str):
    module = types.ModuleType(name)
    exec(code, module.__dict__)
    return module


if __name__ == '__main__':

    code = """
def hello_world():
    print('hello world!')
"""

    my_module = to_module(code, 'mymod')
    for var, val in vars(my_module).items():
        if callable(val):
            print(var, val)

