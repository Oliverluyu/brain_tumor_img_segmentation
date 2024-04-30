import numpy as np
import os
import collections
import json
from models.AttentionUnet import unet_CT_single_att
from models.VanillaUnet import unet_2D
from models.MultiheadAttentionUnet import MultiheadAttentionUnet
from models.CMUnet import CMUnet

from models.CMUnet_with_msag import CMUnet_msag
from models.CMUNeXt import CMUNeXt
from models.ProposedUnet import ProposedAttentionUnet



def json_to_py_obj(filename):
    def _json_object_hook(d): return collections.namedtuple('X', d.keys())(*d.values())
    def json_to_obj(data): return json.loads(data, object_hook=_json_object_hook)
    return json_to_obj(open(filename).read())


def get_model(name):

    return {
        'ProposedUnet': ProposedAttentionUnet,
        'AttentionUnet': unet_CT_single_att,
        'VanillaUnet': unet_2D,
        'MultiheadAttentionUnet': MultiheadAttentionUnet,
        'CMUnet':CMUnet,
        'CMUnet_with_msag':CMUnet_msag,
        'CMUNeXt':CMUNeXt,
    }[name]
