import os
import ltr.admin.loading as ltr_loading
from pytracking.evaluation.environment import env_settings


def load_network(net_path):
    if os.path.isabs(net_path):
        path_full = net_path
        net, _ = ltr_loading.load_network(path_full, backbone_pretrained=False)

    elif isinstance(env_settings().network_path, (list, tuple)):
        net = None
        for p in env_settings().network_path:
            path_full = os.path.join(p, net_path)
            try:
                net, _ = ltr_loading.load_network(path_full, backbone_pretrained=False)
                break
            except:
                pass

        assert net is not None, 'Failed to load network'
    else:
        path_full = os.path.join(env_settings().network_path, net_path)
        net, _ = ltr_loading.load_network(path_full, backbone_pretrained=False)

    return net

def load_network_zpy(net_path1, net_path2):
    
    path_full1 = os.path.join(env_settings().network_path, net_path1)
    path_full2 = os.path.join(env_settings().network_path, net_path2)
    # net, _ = ltr_loading.load_network(path_full1)
    net = ltr_loading.load_network_new(path_full1, path_full2)

    return net
