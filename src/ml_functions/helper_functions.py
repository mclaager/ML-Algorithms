"""
Contains helpful functions that provide additional functionality,
not essential for neural network functionality
"""

# https://stackoverflow.com/a/62662138/11319692
def reraise(e, *args):
  '''re-raise an exception with extra arguments
  :param e: The exception to reraise
  :param args: Extra args to add to the exception
  '''

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