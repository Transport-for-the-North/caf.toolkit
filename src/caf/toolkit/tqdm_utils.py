# -*- coding: utf-8 -*-
"""A collection of utilities to manage tqdm write outs to terminal."""
# Built-Ins
import sys
import contextlib

# Third Party
from tqdm import contrib as tqdm_contrib

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #

# # # CLASSES # # #


# # # FUNCTIONS # # #
@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    """Redirect stdout and stderr to `tqdm.write`.

    Code copied from tqdm documentation:
    https://github.com/tqdm/tqdm#redirecting-writing

    Redirect stdout and stderr to tqdm allows tqdm to control
    how print statements are shown and stops the progress bar
    formatting from breaking. Note: warnings.warn() messages
    still cause formatting issues in terminal.

    Yields
    ------
    sys.stdout:
        Original stdout.

    Examples
    --------
    To use with tqdm, call like this:
    >>> from tqdm import tqdm
    >>> from time import sleep
    >>>
    >>> # Redirect stdout to tqdm.write() (don't forget the `as save_stdout`)
    >>> with std_out_err_redirect_tqdm() as orig_stdout:
    >>> # tqdm needs the original stdout
    >>> # and dynamic_ncols=True to autodetect console width
    >>> for i in tqdm(range(3), file=orig_stdout, dynamic_ncols=True):
    >>>     sleep(.5)
    """
    # Init
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(tqdm_contrib.DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]

    # Relay exceptions
    except Exception as exc:
        raise exc

    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err
