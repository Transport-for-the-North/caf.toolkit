# -*- coding: utf-8 -*-
"""Constant types for use in CAF.Toolkit."""
# Built-Ins

from typing import Literal

# Third Party

# Local Imports
# pylint: disable=import-error,wrong-import-position

# pylint: enable=import-error,wrong-import-position

# # # CONSTANTS # # #
SparseLiteral = Literal["disallow", "allow", "force", "feasible"]
WarningActionKind = Literal["default", "error", "ignore", "always", "module", "once"]

# # # CLASSES # # #

# # # FUNCTIONS # # #
