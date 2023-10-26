.. caf.toolkit documentation master file, created by
   sphinx-quickstart on Wed Oct  4 13:40:35 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to caf.toolkit's documentation!
=======================================

The Common Analytical Framework (CAF) is a collection of transport planning and appraisal functionalities. It’s part of a project to make useful transport related functionality more widely available and easily accessible.

Tool Info
---------
CAF.Toolkit focusses on generic tools and functions that are utilised across the CAF Framework. Over time, alongside adding extras functionality to enhance other CAF packages, CAF.Toolkit will be populated with much of the generic functionality found in `NorMITs Demand <https://github.com/Transport-for-the-North/NorMITs-Demand>`_.

Installation
------------
caf.toolkit can be installed either from pip or conda forge:

``pip install caf.toolkit``

``conda install caf.toolkit -c conda-forge``



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

Modules
--------

.. toctree::
   :maxdepth: 1

   array_utils
   config_base
   cost_utils
   io
   iterative_proportional_fitting
   log_helpers
   math_utils
   timing
   toolbox
   tqdm_utils
   translation
   validators

Sub-packages
------------

.. toctree::
   :maxdepth: 1

   concurrency
   pandas_utils
