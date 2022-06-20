import os
import glob
import torch
from ltr.admin import loading


class BaseTrainer:
    """Base trainer class. Contains functions for training and saving/loading chackpoints.
    Trainer classes should inherit from this one and overload the train_epoch function."""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        args:
            actor - The actor for training the network
            loaders - list of dataset loaders, e.g. [train_loader, val_loader]. In each epoch, the trainer runs one
                        epoch for each loader.
            optimizer - The optimizer used for training, e.g. Adam
            settings - Training settings
            lr_scheduler - Learning rate scheduler
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders

        self.update_settings(settings)

        self.epoch = 0
        self.stats = {}

        self.device = getattr(settings, 'device', None)
        if self.device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and settings.use_gpu else "cpu")

        self.actor.to(self.device)

    def update_settings(self, settings=None):
        """Updates the trainer settings. Must be called to update internal settings."""
        if settings is not None:
            self.settings = settings

        if self.settings.env.workspace_dir is not None:
            self.settings.env.workspace_dir = os.path.expanduser(self.settings.env.workspace_dir)
            self._checkpoint_dir = os.path.join(self.settings.env.workspace_dir, 'checkpoints')
            if not os.path.exists(self._checkpoint_dir):
                os.makedirs(self._checkpoint_dir)
        else:
            self._checkpoint_dir = None


    def train(self, max_epochs, load_latest=False, fail_safe=True):
        """Do training for the given number of epochs.
        args:
            max_epochs - Max number of training epochs,
            load_latest - Bool indicating whether to resume from latest epoch.
            fail_safe - Bool indicating whether the training to automatically restart in case of any crashes.
        """

        epoch = -1
        num_tries = 10
        for i in range(num_tries):
            # try:
                if load_latest:
                    self.load_checkpoint()
                    # self.epoch = 0
                for epoch in range(self.epoch+1, max_epochs+1):
                    self.epoch = epoch

                    if self.lr_scheduler is not None:
                        self.lr_scheduler.step()

                    self.train_epoch()

                    if epoch % 1 == 0:
                        if self._checkpoint_dir:
                            self.save_checkpoint()
            # except:
            #     print('Training crashed at epoch {}'.format(epoch))
            #     if fail_safe:
            #         load_latest = True
            #         print('Restarting training from last epoch ...')
            #     else:
            #         raise

        print('Finished training!')


    def train_epoch(self):
        raise NotImplementedError


    def save_checkpoint(self):
        """Saves a checkpoint of the network and other variables."""

        actor_type = type(self.actor).__name__
        net_type = type(self.actor.net).__name__
        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': self.actor.net.state_dict(),
            'net_info': getattr(self.actor.net, 'info', None),
            'constructor': getattr(self.actor.net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings
        }


        directory = '{}/{}'.format(self._checkpoint_dir, self.settings.project_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        file_path = '{}/{}_ep{:04d}.pth.tar'.format(directory, net_type, self.epoch)
        torch.save(state, file_path)


    def load_checkpoint(self, checkpoint = None, fields = None, ignore_fields = None, load_constructor = False):
        """Loads a network checkpoint file.

        Can be called in three different ways:
            load_checkpoint():
                Loads the latest epoch from the workspace. Use this to continue training.
            load_checkpoint(epoch_num):
                Loads the network at the given epoch number (int).
            load_checkpoint(path_to_checkpoint):
                Loads the file from the given absolute path (str).
        """

        actor_type = type(self.actor).__name__
        net_type = type(self.actor.net).__name__

        if checkpoint is None:
            # Load most recent checkpoint
            checkpoint_list = sorted(glob.glob('{}/{}/{}_ep*.pth.tar'.format(self._checkpoint_dir,
                                                                             self.settings.project_path, net_type)))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return
        elif isinstance(checkpoint, int):
            # Checkpoint is the epoch number
            checkpoint_path = '{}/{}/{}_ep{:04d}.pth.tar'.format(self._checkpoint_dir, self.settings.project_path,
                                                                 net_type, checkpoint)
        elif isinstance(checkpoint, str):
            # checkpoint is the path
            checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError

        # Load network
        checkpoint_dict = loading.torch_load_legacy(checkpoint_path)

        assert net_type == checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                self.actor.net.load_state_dict(checkpoint_dict[key], strict = False)
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            self.actor.net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            self.actor.net.info = checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True

    def load_checkpoint_new(self, checkpoint1 = None, checkpoint2 = None, fields = None, ignore_fields = None, load_constructor = False):
        

        actor_type = type(self.actor).__name__
        net_type = type(self.actor.net).__name__

        # Load network
        dis_checkpoint_dict = loading.torch_load_legacy(checkpoint1)
        comp_checkpoint_dict = loading.torch_load_legacy(checkpoint2)

        assert net_type == dis_checkpoint_dict['net_type'], 'Network is not of correct type.'

        if fields is None:
            fields = dis_checkpoint_dict.keys()
        if ignore_fields is None:
            ignore_fields = ['settings']

            # Never load the scheduler. It exists in older checkpoints.
        ignore_fields.extend(['lr_scheduler', 'constructor', 'net_type', 'actor_type', 'net_info'])

        # Load all fields
        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                pretrain_dict = {k[len('feature_extractor.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_extractor.'):] in self.actor.net.feature_extractor.state_dict()}
                self.actor.net.feature_extractor.load_state_dict(pretrain_dict)

                pretrain_dict = {k[len('feature_extractor_i.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_extractor_i.'):] in self.actor.net.feature_extractor_i.state_dict()}
                self.actor.net.feature_extractor_i.load_state_dict(pretrain_dict)

                pretrain_dict = {k[len('classifier.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('classifier.'):] in self.actor.net.classifier.state_dict()}
                self.actor.net.classifier.load_state_dict(pretrain_dict)

                pretrain_dict = {k[len('feature_fusion.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('feature_fusion.'):] in self.actor.net.feature_fusion.state_dict()}
                self.actor.net.feature_fusion.load_state_dict(pretrain_dict)
                
                # pretrain_dict = {k[len('bb_regressor.'):]: v for k, v in dis_checkpoint_dict['net'].items() if k[len('bb_regressor.'):] in self.actor.net.bb_regressor.state_dict()}
                # net.bb_regressor.load_state_dict(pretrain_dict)

                pretrain_dict = {k[len('feature_extractor_vi.'):]: v for k, v in comp_checkpoint_dict['net'].items() if k[len('feature_extractor_vi.'):] in self.actor.net.feature_extractor_vi.state_dict()}
                self.actor.net.feature_extractor_vi.load_state_dict(pretrain_dict)

                pretrain_dict = {k[len('classifier_vi.'):]: v for k, v in comp_checkpoint_dict['net'].items() if k[len('classifier_vi.'):] in self.actor.net.classifier_vi.state_dict()}
                self.actor.net.classifier_vi.load_state_dict(pretrain_dict)

            elif key == 'optimizer':
                continue
                # self.optimizer.load_state_dict(dis_checkpoint_dict[key])
            else:
                setattr(self, key, dis_checkpoint_dict[key])

        # Set the net info
        if load_constructor and 'constructor' in dis_checkpoint_dict and dis_checkpoint_dict['constructor'] is not None:
            self.actor.net.constructor = dis_checkpoint_dict['constructor']
        if 'net_info' in dis_checkpoint_dict and dis_checkpoint_dict['net_info'] is not None:
            self.actor.net.info = dis_checkpoint_dict['net_info']

        # Update the epoch in lr scheduler
        if 'epoch' in fields:
            self.lr_scheduler.last_epoch = self.epoch

        return True



