import os
import sys
from importlib import import_module
from pathlib import Path
import yaml
import fileinput

import comet_ml

from mmodel.basic_params import basic_parser


# 将basic_parser所有的参数全部设置好
def get_basic_params():
    model = os.environ["TARGET_MODEL"]
    config_path = Path('./mmodel/{}/config.yml'.format(model))
    if not config_path.exists():
        raise Exception("{} should be config file".format(config_path))
    
    save_path = Path('./mmodel/{}/__saved_model__'.format(model))
    save_path.mkdir(exist_ok=True)
    os.environ['SVAE_PATH'] = str(save_path)
    
    log_path = Path('./mmodel/{}/__log__'.format(model))
    log_path.mkdir(exist_ok=True)
    os.environ['SVAE_LOG'] = str(log_path)

    basic_parser._default_config_files.append(config_path)
    params, _ = basic_parser.parse_known_args()
    return params

# 生成model
def get_model():
    model_name = os.environ["TARGET_MODEL"]

    dir_path = Path('./mmodel/'+model_name)
    if not dir_path.exists():
        raise Exception("{} not exists.".format(model_name))
    os.environ["TARGET_MODEL"] = model_name
    #print("qwuteiqtweiqw ")
    model = import_module('mmodel.' + model_name).model()
    #print(model)
    return model

