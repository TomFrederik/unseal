.. _hooking:


===============
Hooking
===============

Hooks are at the heart of Unseal. In short, a hook is an access point to a model. It is defined by the point of the model at 
which it attaches and by the operation that it executes (usually either during the forward or backward pass).

To read more about the original concept of a hook in PyTorch read `here <https://pytorch.org/docs/stable/notes/modules.html#module-hooks>`_.

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
    # print(model.structure['module'])

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

The most important feature of a HookedModel object is its modified ``forward`` method which allows a user to temporarily add a hook to the model, perform a forward pass
and record the result in the context attribute of the HookedModel.

For this, the forward method takes an additional ``hooks`` argument which is a ``list`` of ``Hooks`` which get registered. After the forward pass, the hooks are removed
again (to ensure consistent behavior). Hooks have access to the ``save_ctx`` attribute of the HookedModel, so anything you want to access later goes there and can
be recalled via ``model.save_ctx[your_hook_key]``


Writing hooks
-----------------

As mentioned above, hooks are triples ``(layer_name, func, key)``. After choosing the attachment point (the ``layer_name``, an element from ``model.layers.keys()``), 
you need to implement the hooking function. 

Every hooking function needs to follow the signature ``save_ctx, input, output -> output``. 

``save_ctx`` is a dictionary which is initialized empty by the HookedModule class
during the forward pass. ``input`` and ``output`` are the input and output of the module respectively. If the hook is not modifying the output, the function does
not need to return anything, as that is the default behavior.

For example, let's implement a hook which saves the input and output to first linear layer in the network we defined above:


.. code-block:: python

    import torch
    from unseal import Hook

    # make sure to not clutter the gpu and not keep track of gradients.
    def save_input_output(save_ctx, input, output):
        save_ctx['input'] = input.detach().cpu()
        save_ctx['output'] = output.detach().cpu()
    
    input_tensor = torch.rand((1,8))
    my_hook = Hook('0', func, 'save_input_output_0')
    
    model.forward(input_tensor, hooks=[my_hook])

    # now we can access the model's context object
    print(model.save_ctx['save_input_output_0']['input'])
    print(model.save_ctx['save_input_output_0']['output'])
    
    '''Output:
    tensor([[0.5778, 0.0257, 0.4552, 0.4787, 0.9211, 0.0284, 0.8347, 0.9621]])
    tensor([[-0.6566,  1.0794,  0.1455, -0.0396,  0.0411,  0.2184, -0.3484, -0.1095,
            -0.2990, -0.1757,  0.1078,  0.2126,  0.4414,  0.1682, -0.2449,  0.0090,
            -0.0726, -0.0325, -0.5832,  0.1020, -0.2699,  0.0223, -0.8340, -0.4016,
            -0.2808, -0.5337,  0.1518,  1.1230,  1.1380, -0.1437,  0.2738,  0.4592,
            -0.7136, -0.3247,  0.2068, -0.5012,  0.4446, -0.4551,  0.2015, -0.3641,
            -0.1598, -0.7272,  0.0271,  0.2181, -0.3253,  0.2763, -0.5745,  0.4344,
            0.0255, -0.2492,  0.1586,  0.2404, -0.2033, -0.6197, -0.1098,  0.3736,
            0.1246, -0.4697, -0.7690,  0.0981, -0.0255,  0.2133,  0.3061,  0.1846]])
    '''