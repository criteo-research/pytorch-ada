Known issues
############


AttributeError: 'MultiDataLoader' object has no attribute 'sampler':
--------------------------------------------------------------------

This bug has been fixed in pytorch-lightning only recently and appears only in the multi-gpu mode. Quickest fix (for now) is to use a single GPU or revert to version 0.7.5 with ``pip install pytorch-lightning==0.7.5``.


TypeError: 'NoneType' object is not iterable
--------------------------------------------
If this error appears with multiple traces, involving also

    ...
    TypeError: forward() takes 2 positional arguments but 3 were given
    Exception ignored in: <object repr() failed>`
    ... errors in tqdm

The current fix is to revert pytorch-lightning to version 0.7.3.
