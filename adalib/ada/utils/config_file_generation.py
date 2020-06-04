import json
import itertools
import copy
import collections


class Iter:
    def __init__(self, *params):
        if len(params) == 1:
            self.iterable = params[0]
        else:
            self.iterable = params


class ConfigVariants:
    def __init__(self):
        self._variant_by_names = dict()
        self._num_duplicated_by_base_names = collections.Counter()

    def add(self, name, **params):
        keys, values = zip(
            *sorted(
                (k, v.iterable if isinstance(v, Iter) else [v])
                for k, v in params.items()
            )
        )
        all_values = list(itertools.product(*values))
        if len(all_values) == 0:
            raise ValueError("Encountered an empty range")

        for values in all_values:
            variant_params = dict(zip(keys, values))
            variant_name = name.format(**variant_params)
            num = self._num_duplicated_by_base_names[variant_name]
            self._num_duplicated_by_base_names[variant_name] += 1
            if num > 0:
                if num == 1:
                    self._variant_by_names[
                        f"{variant_name}#0"
                    ] = self._variant_by_names[variant_name]
                    del self._variant_by_names[variant_name]
                variant_name += f"#{num}"
            self._variant_by_names[variant_name] = variant_params
        return self

    def to_dict(self):
        return copy.deepcopy(self._variant_by_names)

    def save(self, filename):
        with open(filename, "w") as f_out:
            json.dump(self._variant_by_names, f_out, indent=4, sort_keys=True)
