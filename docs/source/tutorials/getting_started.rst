.. _getting_started:

Getting started
===============

Prerequisites
-------------

Unseal requires python 3.6+.


Installation
------------

For its interfaces, Unseal uses `this fork <https://github.com/TomFrederik/pysvelte>`_ of the PySvelte library, which can be installed via pip:

.. code-block:: console

    git clone git@github.com:TomFrederik/PySvelte.git
    cd PySvelte
    pip install -e .


Install Unseal via pip after cloning the repository:

.. code-block:: console
    
    git clone git@github.com:TomFrederik/unseal.git
    cd unseal
    pip install -e .


Using Unseal
------------

If you just want to get started and play around with models, then head to the :ref:`interface <interfaces>` section.

If you want to learn more about Unseal works under the hood, check out our section on :ref:`hooking <hooking>`.