import numpy as np
import os
import collections
import json
from models.AttentionUnet import unet_CT_single_att
from models.VanillaUnet import unet_2D



def json_to_py_obj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json_to_obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json_to_obj(open(filename).read())


def get_model(name):

    return {
        'AttentionUnet': unet_CT_single_att,
        'VanillaUnet': unet_2D
    }[name]
