.. _hooking:


===============
Hooking
===============

Hooks are at the heart of Unseal. In short, a hook is an access point to a model. It is defined by the point of the model at 
which it attaches and by the operation that it executes (usually either during the forward or backward pass).

To read more about the original concept of a hook in PyTorch read `here<https://pytorch.org/docs/stable/notes/modules.html#module-hooks>`_.

In Unseal, a hook is defined as a ``namedtuple`` consisting of a ``layer_name`` (the point at which it attaches), 
a ``func`` (the function it executes), and a ``key`` (an identifying string unique to the hook)

In order to simplify the hooking interface, Unseal wraps every model in the ``hooks.HookedModel`` class. 



hooks.HookedModel
=======================

You can access the top-level structure of a so-wrapped model by printing it (i.e. its ``__repr__`` property) or its ``structure['module']`` property:

.. code-block:: python

    import torch.nn as nn
    from unseal.hooks import HookedModel

    model = nn.Sequential(
        nn.Linear(8,64),
        nn.ReLU(),
        nn.Sequential(
            nn.Linear(64,256),
            nn.ReLU(),
            nn.Linear(256,1)
        )
    )
    model = HookedModel(model)

    print(model)

    # equivalent:
    print(model.structure['module'])

    ''' Output:
    Sequential(
        (0): Linear(in_features=8, out_features=64, bias=True)
        (1): ReLU()
        (2): Sequential(
            (0): Linear(in_features=64, out_features=256, bias=True)
            (1): ReLU()
            (2): Linear(in_features=256, out_features=1, bias=True)
        )
    )
    '''


A HookedModel also has special references to every layer which you can access via the ``layers`` attribute:

.. code-block:: python

    print(model.layers)
    '''Output:
    OrderedDict([('0', Linear(in_features=8, out_features=64, bias=True)), ('1', ReLU()), ('2', Sequential(
        (0): Linear(in_features=64, out_features=256, bias=True)
        (1): ReLU()
        (2): Linear(in_features=256, out_features=1, bias=True)
    )), ('2->0', Linear(in_features=64, out_features=256, bias=True)), ('2->1', ReLU()), ('2->2', Linear(in_features=256, out_features=1, bias=True))])
    '''


You can see that each layer has its own identifying string (e.g. ``'2->2'``). If you want to only display the layer names you can simply call ``model.layers.keys()``.

forward passes
--------------

The most important feature of a HookedModel object is its modified forward method which allows a user to temporarily add a hook to the model, perform a forward pass
and record the result in the context attribute of the HookedModel.

For this, the forward method takes an additional ``hooks`` argument which is a ``list`` of ``Hooks`` which get registered. After the forward pass, the hooks are removed
again (to ensure consistent behavior). Hooks have access to the ``save_ctx`` attribute of the HookedModel, so anything you want to access later goes there and can
be recalled via ``model.save_ctx[your_hook_key]``


Writing hooks
-----------------

As mentioned above, hooks are a triple ``(layer_name, func, key)``. After choosing the attachment point (the ``layer_name``, an element from ``model.layers.keys()``), 
you need to implement the hooking function. 

Every hooking function needs to follow the signature ``save_ctx, input, output -> output``. 

``save_ctx`` is a dictionary which is initialized empty by the HookedModule class
during the forward pass. ``input`` and ``output`` are the input and output of the module respectively. If the hook is not modifying the output, the function does
not need to return anything, as that is the default behavior.