import os
import sys
import multiprocessing

env_path = os.path.join(os.path.dirname(__file__), '..')
if env_path not in sys.path:
    sys.path.append(env_path)

from ltr.training import run_training
os.environ["CUDA_VISIBLE_DEVICES"] ="0"

import warnings
warnings.simplefilter("ignore",UserWarning)

def main():
    train_module = 'seq_tracking'
    train_name = 'train_RGBT'

    run_training(train_module, train_name)


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn', force=True)
    main()
