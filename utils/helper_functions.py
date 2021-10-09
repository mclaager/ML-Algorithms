"""
Contains helpful functions that provide additional functionality,
not essential for neural network functionality
"""

import numpy as np

# https://stackoverflow.com/a/62662138/11319692
def reraise(e, *args):
    """
    re-raise an exception with extra arguments
    
    :param e: The exception to reraise
    :param args: Extra args to add to the exception
    """

    # e.args is a tuple of arguments that the exception with instantiated with.
    e.args = args + e.args

    # Recreate the expection and preserve the traceback info so that we can see 
    # where this exception originated.
    raise e.with_traceback(e.__traceback__)

def str_or_func(module, identifier, err_msg = ''):
    """
    If string is inputted, the function will be searched for in
    the given module. Otherwise, the function itself will be returned.

    :param module: The module that will be searched
    :param identifier: String or Func that will be searched
    :param err_msg: A custom error message if identifier is not in module

    :raise AttributeError: Given string was not found in module
    """

    if isinstance(identifier, str):
      try:
        return getattr(module, identifier)
      except AttributeError as err:
        reraise(err, err_msg)
    else:
        return identifier

def str_or_class(module, identifier, err_msg = '', **kwargs):
    """
    If string is inputted, the class will be searched for in
    the given module and a no-param instance returned.
    Otherwise, the instance itself will be returned.

    :param module: The module that will be searched
    :param identifier: String or Class that will be searched 
    :param err_msg: A custom error message if identifier is not in module
    :param **kwargs: A list of agruments to instantiate class with

    :raise AttributeError: Given string was not found in module
    """

    if isinstance(identifier, str):
      try:
        return getattr(module, identifier)(**kwargs)
      except AttributeError as err:
        reraise(err, err_msg)
    else:
        return identifier

def coerce_1d_array(arr: np.ndarray or list, new_dimensions: int = 2, axis: int = 0) -> np.ndarray:
    """
    If a 1D numpy array is given, it will returned a coerced nD array with the original values
    along the chosen axis.

    :param arr: The array to be coerced
    :param new_dimensions: The number of dimensions the coerced array should be
    :param axis: The axis to put the original values

    :returns: The original array or a nD coersion of that array
    """
    new_arr = np.array(arr)
    if len(new_arr.shape) == 1:
        new_axis = [-1 if s == axis else 1 for s in range(new_dimensions)]
        new_arr = new_arr.reshape(tuple(new_axis))
    return new_arr