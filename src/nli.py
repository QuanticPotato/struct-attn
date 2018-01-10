import PyTorch
import PyTorchHelpers
import os.path

data = PyTorchHelpers.load_lua_class(os.path.join('..', 'data-entail.lua'), 'data')

print(data)