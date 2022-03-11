.. _installing_unseal:

=====================
Installing Unseal
=====================

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

In order to run PySvelte, you will also need to install ``npm`` via your package manager.

Install Unseal via pip after cloning the repository:

.. code-block:: console
    
    git clone git@github.com:TomFrederik/unseal.git
    cd unseal
    pip install -e .