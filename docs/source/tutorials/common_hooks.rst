.. _common_hooks:

============================
Common Hooks
============================

The most common hooking functions are supported out of the box by Unseal.

Some of the methods can be used directly as a function in a hook, others return such a function and some will return the hook itself.
This will be indicated in the docstring.

Saving Outputs
==============

This method can be used directly in the construction of a hook.

.. automethod:: unseal.hooks.common_hooks.save_output


Replacing Activations
=====================

This method is a factory and returns a function that can be used in a hook to replace the activation of a layer.

.. automethod:: unseal.hooks.common_hooks.replace_activation


Saving Attention
=====================

.. automethod:: unseal.hooks.common_hooks.transformers_get_attention


Creating an Attention Hook
===========================

.. automethod:: unseal.hooks.common_hooks.create_attention_hook


Creating a Logit Hook
===========================

.. automethod:: unseal.hooks.common_hooks.create_logit_hook


GPT ``_attn`` Wrapper
=====================

.. automethod:: unseal.hooks.common_hooks.gpt_attn_wrapper


