import torch
import os
import sys
from pathlib import Path
import importlib


def load_network(network_dir=None, checkpoint=None, constructor_fun_name=None, constructor_module=None, **kwargs):
        """Loads a network checkpoint file.

        Can be called in two different ways:
            load_checkpoint(network_dir):
                Loads the checkpoint file given by the path. I checkpoint_dir is a directory,
                it tries to find the latest checkpoint in that directory.
            load_checkpoint(network_dir, checkpoint=epoch_num):
                Loads the network at the given epoch number (int).

        The extra keyword arguments are supplied to the network constructor to replace saved ones.
        """


        if network_dir is not None:
            net_path = Path(network_dir)
        else:
            net_path = None

        if net_path.is_file():
            checkpoint = str(net_path)

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(net_path.glob('*.pth.tar'))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                raise Exception('No matching checkpoint file found')
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_list = sorted(net_path.glob('*_ep{:04d}.pth.tar'.format(checkpoint)))
            if not checkpoint_list or len(checkpoint_list) == 0:
                raise Exception('No matching checkpoint file found')
            if len(checkpoint_list) > 1:
                raise Exception('Multiple matching checkpoint files found')
            else:
                checkpoint_path = checkpoint_list[0]
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = torch_load_legacy(checkpoint_path)
        # print(checkpoint_dict['constructor'])
        # Construct network model
        if 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net_constr = checkpoint_dict['constructor']
            if constructor_fun_name is not None:
                net_constr.fun_name = constructor_fun_name
            if constructor_module is not None:
                net_constr.fun_module = constructor_module
            for arg, val in kwargs.items():
                if arg in net_constr.kwds.keys():
                    net_constr.kwds[arg] = val
                else:
                    print('WARNING: Keyword argument "{}" not found when loading network.'.format(arg))
            # Legacy networks before refactoring
            if net_constr.fun_module.startswith('dlframework.'):
                net_constr.fun_module = net_constr.fun_module[len('dlframework.'):]
            net = net_constr.get()
        else:
            raise RuntimeError('No constructor for the given network.')

        net.load_state_dict(checkpoint_dict['net'])

        net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        return net, checkpoint_dict

def load_network_new(dis_network_dir, comp_network_dir):

    dis_checkpoint_dict = torch_load_legacy(dis_network_dir)
    comp_checkpoint_dict = torch_load_legacy(comp_network_dir)

    net_constr = dis_checkpoint_dict['constructor']
    net = net_constr.get()
    # Construct network model
    pretrain_dict = {k[len('feature_extractor.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_extractor.'):] in net.feature_extractor.state_dict()}
    net.feature_extractor.load_state_dict(pretrain_dict)

    pretrain_dict = {k[len('feature_extractor_i.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_extractor_i.'):] in net.feature_extractor_i.state_dict()}
    net.feature_extractor_i.load_state_dict(pretrain_dict)

    pretrain_dict = {k[len('classifier.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('classifier.'):] in net.classifier.state_dict()}
    net.classifier.load_state_dict(pretrain_dict)

    pretrain_dict = {k[len('feature_fusion.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_fusion.'):] in net.feature_fusion.state_dict()}
    net.feature_fusion.load_state_dict(pretrain_dict)
    
    pretrain_dict = {k[len('bb_regressor.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('bb_regressor.'):] in net.bb_regressor.state_dict()}
    net.bb_regressor.load_state_dict(pretrain_dict)

    pretrain_dict = {k[len('feature_extractor_vi.'):]: v for k, v in comp_checkpoint_dict['net'].items() if k[len('feature_extractor_vi.'):] in net.feature_extractor_vi.state_dict()}
    net.feature_extractor_vi.load_state_dict(pretrain_dict)

    pretrain_dict = {k[len('classifier_vi.'):]: v for k, v in comp_checkpoint_dict['net'].items() if k[len('classifier_vi.'):] in net.classifier_vi.state_dict()}
    net.classifier_vi.load_state_dict(pretrain_dict)


    net.constructor = dis_checkpoint_dict['constructor']


    return net

def load_weights(net, path, strict=True):
    checkpoint_dict = torch.load(path)
    weight_dict = checkpoint_dict['net']
    net.load_state_dict(weight_dict, strict=strict)
    return net


def torch_load_legacy(path):
    """Load network with legacy environment."""

    # Setup legacy env (for older networks)
    _setup_legacy_env()

    # Load network
    checkpoint_dict = torch.load(path)

    # Cleanup legacy
    _cleanup_legacy_env()

    return checkpoint_dict


def _setup_legacy_env():
    importlib.import_module('ltr')
    sys.modules['dlframework'] = sys.modules['ltr']
    sys.modules['dlframework.common'] = sys.modules['ltr']
    for m in ('model_constructor', 'stats', 'settings', 'local'):
        importlib.import_module('ltr.admin.'+m)
        sys.modules['dlframework.common.utils.'+m] = sys.modules['ltr.admin.'+m]


def _cleanup_legacy_env():
    del_modules = []
    for m in sys.modules.keys():
        if m.startswith('dlframework'):
            del_modules.append(m)
    for m in del_modules:
        del sys.modules[m]
