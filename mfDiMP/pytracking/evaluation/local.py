from pytracking.evaluation.environment import EnvSettings
import os 

def local_env_settings():
    settings = EnvSettings()

    cur_dir = os.getcwd()

    settings.results_path = None
    settings.network_path = os.path.abspath(os.path.join(cur_dir,'../..'))

    return settings

