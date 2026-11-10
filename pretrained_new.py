# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch

from esmfold_new import ESMFold


def _load_model(model_name):
    url = "https://dl.fbaipublicfiles.com/fair-esm/models/esmfold_structure_module_only_650M.pt"
    model_data = torch.hub.load_state_dict_from_url(url, progress=False, map_location="cpu")
    cfg = model_data["cfg"]["model"]
    
    model_state = torch.load(str(model_name), map_location="cpu")
    model = ESMFold(esmfold_config=cfg)

    expected_keys = set(model.state_dict().keys())
    found_keys = set(model_state.keys())

    missing_essential_keys = []
    for missing_key in expected_keys - found_keys:
        if not missing_key.startswith("esm."):
            missing_essential_keys.append(missing_key)

    if missing_essential_keys:
        raise RuntimeError(f"Keys '{', '.join(missing_essential_keys)}' are missing.")

    model.load_state_dict(model_state, strict=False)

    return model


def esmfold_v0():
    return _load_model("esmfold_3B_v0")


def esmfold_v1():
    return _load_model("esmfold_3B_v1")


def esmfold_structure_module_only_8M():
    return _load_model("esmfold_structure_module_only_8M")


def esmfold_structure_module_only_8M_270K():
    return _load_model("esmfold_structure_module_only_8M_270K")


def esmfold_structure_module_only_35M():
    return _load_model("esmfold_structure_module_only_35M")


def esmfold_structure_module_only_35M_270K():
    return _load_model("esmfold_structure_module_only_35M_270K")


def esmfold_structure_module_only_150M():
    return _load_model("esmfold_structure_module_only_150M")


def esmfold_structure_module_only_150M_270K():
    return _load_model("esmfold_structure_module_only_150M_270K")


def esmfold_structure_module_only_650M():
    return _load_model("esmfold_structure_module_only_650M")


def esmfold_structure_module_only_650M_270K():
    return _load_model("esmfold_structure_module_only_650M_270K")


def esmfold_structure_module_only_3B():
    return _load_model("esmfold_structure_module_only_3B")


def esmfold_structure_module_only_3B_270K():
    return _load_model("esmfold_structure_module_only_3B_270K")


def esmfold_structure_module_only_15B():
    return _load_model("esmfold_structure_module_only_15B")
