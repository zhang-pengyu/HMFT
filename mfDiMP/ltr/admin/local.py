import os 
class EnvironmentSettings:
    def __init__(self):
        cur_dir = os.getcwd()
        self.workspace_dir = os.path.abspath(os.path.join(cur_dir,'../networks/'))   # Base directory for saving network checkpoints.
        self.tensorboard_dir = self.workspace_dir + '/tensorboard/'    # Directory for tensorboard files.
        self.UAV_RGBT_dir = '' # path for training dataset
