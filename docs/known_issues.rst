Known issues
############


 AttributeError: 'MultiDataLoader' object has no attribute 'sampler'
 
 This bug has been fixed in pytorch-lightning only recently and appears only in the multi-gpu mode. Quickest fix (for now) is to use a single GPU.

