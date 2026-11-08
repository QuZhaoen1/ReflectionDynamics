"""Keep mixed-dtype ZeRO parameters in separate gather calls, without casting."""
from collections import OrderedDict


def group_by_partition_dtype(params):
    groups = OrderedDict()
    for param in params:
        groups.setdefault(param.ds_tensor.dtype, []).append(param)
    return list(groups.values())


def install():
    import inspect
    from deepspeed.runtime.zero.partition_parameters import Init
    original = Init._allgather_params_coalesced
    if getattr(original, '_rd_dtype_fix', False):
        return
    if 'dtype=param_list[0].ds_tensor.dtype' not in inspect.getsource(original):
        raise RuntimeError('DeepSpeed gather implementation changed; review compatibility patch')

    def grouped(self, param_list, hierarchy=0, quantize=False):
        groups = group_by_partition_dtype(param_list)
        if len(groups) <= 1 or quantize:
            return original(self, param_list, hierarchy=hierarchy, quantize=quantize)
        # Original call is blocking. Each group retains its parameter order,
        # partition dtype, communication group, and original gather arithmetic.
        for params in groups:
            original(self, params, hierarchy=hierarchy, quantize=False)
        return None

    grouped._rd_dtype_fix = True
    Init._allgather_params_coalesced = grouped
