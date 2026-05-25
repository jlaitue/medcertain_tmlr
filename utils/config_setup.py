import warnings

warnings.filterwarnings(
    "ignore",
    message="distutils Version classes are deprecated.*",
    category=DeprecationWarning,
)
import os
import json
import numpy as np
import random as random_py
from utils.arguments import args_parser
from utils.constants import NUM_CLASSES
from MedFuse.models import lstm_jax, fusion_jax
from utils.base_architectures import ResNetMod, ResNetBlock
from flax import linen as nn

def model_architecture_args(args):
     #-----------------------------------------------------------------------------
    # GENERAL SETUP OF MODEL ARCHITECTURE
    if 'ResNet34' in args["model_name"]:
        model_class = ResNetMod
        num_blocks = (3, 4, 6, 3)
        c_hidden = (64, 128, 256, 512)
    if 'ResNet50' in args["model_name"]:
        model_class = ResNetMod
        num_blocks = None
        c_hidden = (64, 128, 256, 512)
    if 'LSTM' in args["model_name"]:
        model_class = lstm_jax.LSTM
        num_blocks = None
        c_hidden = None
    if 'Fusion' in args["model_name"]:
        model_class = fusion_jax.Fusion
        num_blocks = None
        c_hidden = None

    args["model_class"] = model_class
    args["num_blocks"] = num_blocks
    args["c_hidden"] = c_hidden
    args["act_fn"] = nn.relu
    args["block_class"] = ResNetBlock
    #-----------------------------------------------------------------------------
    # GENERAL SETUP OF METHOD
    if args["prior_precision"] == 0:
        args["prior_precision"] = 1 / args["prior_var"]
    elif args["prior_var"] == 0:
        args["prior_var"] = 1 / args["prior_precision"]
    else:
        raise ValueError("Only one of prior_precision and prior_var can be set.")

    args["prior_mean"] = "Pretrained Mean" if "Pretrained" in args["model_name"] else args["prior_mean"]

    if args["method"] in ["fsmap", "psmap"]:
        args["stochastic"] = False
    if args["method"] in ["fsvi", "psvi"]:
        args["stochastic"] = True
    
    return args

def setup_script():
    # PARSER
    parser = args_parser()
    args = parser.parse_args()
    args_dict = vars(args)

    config_file = args.config
    config_id = args.config_id
    config_name = args.config_name
    cwd = os.getcwd()

    if config_file != '':
        with open(config_file, 'r') as f:
            config_json = json.load(f)

        configurations = config_json['configurations']
        if config_name == '':
            name = configurations[config_id]['name']
        else:
            name = config_name
        id = configurations[config_id]['id']
        cwd = os.getcwd() + "/laboratory/"
        parser_args_list = configurations[config_id]['args']
        env_args = configurations[config_id]['env']

        def is_float(string):
            try:
                float(string)
                return True
            except ValueError:
                return False
            
        parser_args = {}

        for i in range(len(parser_args_list)):
            if parser_args_list[i].startswith('--'):
                key = parser_args_list[i][2:]
                value = parser_args_list[i+1]
                parser_args[key] = value

        print(f"\nCONFIG NAME: {name}")
        print(f"\nWORKING DIRECTORY: {cwd}")
        print(f"\nCONFIG ID: {id}")
        print(f"\nENVIRONMENT ARGS:\n\n{env_args}")

        for key in parser_args:
            args_dict[key] = parser_args[key]

        for key in parser_args:
            if isinstance(parser_args[key], int):
                args_dict[key] = int(parser_args[key])
            elif isinstance(parser_args[key], str) and parser_args[key].isnumeric():
                args_dict[key] = int(parser_args[key])
            elif isinstance(parser_args[key], str) and is_float(parser_args[key]):
                args_dict[key] = float(parser_args[key])
            elif parser_args[key] == 'True' or parser_args[key] == 'False':
                args_dict[key] = True if parser_args[key] == 'True' else False

        for key in env_args:
            os.environ[key] = env_args[key]

    # CXR context dataset hyperparams
    cxr_context_hypers = {
        "randcrop_size": args_dict["cxr_context_randcrop_size"],
        "randhorizontalflip_prob": args_dict["cxr_context_randhorizontalflip_prob"],
        "randverticalflip_prob": args_dict["cxr_context_randverticalflip_prob"],
        "gaussianblur_kernel": args_dict["cxr_context_gaussianblur_kernel"],
        "randsolarize_threshold": args_dict["cxr_context_randsolarize_threshold"],
        "randsolarize_prob": args_dict["cxr_context_randsolarize_prob"],
        "randinvert_prob": args_dict["cxr_context_randinvert_prob"],
        "colorjitter_brightness": args_dict["cxr_context_colorjitter_brightness"],
        "colorjitter_contrast": args_dict["cxr_context_colorjitter_contrast"],
    }

    # EHR context dataset hyperparams
    ehr_context_hypers = {
        "dropstart_max_percent": args_dict["ehr_context_dropstart_max_percent"],
        "gaussian_mean": args_dict["ehr_context_gaussian_mean"],
        "gaussian_std": args_dict["ehr_context_gaussian_std"],
        "gaussian_max_cols": args_dict["ehr_context_gaussian_max_cols"]
    }

    if cwd == "":
        raise Exception("Working directory has not been explicitly set.")
    
    print(f"\nPARSER ARGS:\n\n{args}")
    os.chdir(cwd)
    seed = args_dict["seed"]

    import jax
    import tensorflow as tf
    from jax import random
    import torch
    from jax.config import config

    # Seeding for random operations
    print(f"\nSEED: {seed}")
    main_rng = random.PRNGKey(seed)
    rng_key = main_rng
    os.environ["PYTHONHASHSEED"] = str(seed)
    random_py.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    torch.random.manual_seed(seed)

    print(f"\nCURRENT COMPUTE DEVICE: {jax.devices()[0]}\n")

    if args_dict["debug"]: 
        config.update("jax_disable_jit", True)

    args_dict["cxr_context_hypers"] = cxr_context_hypers
    args_dict["ehr_context_hypers"] = ehr_context_hypers
    args_dict["cwd"] = cwd
    args_dict = model_architecture_args(args_dict)

    args_dict["num_classes"] = NUM_CLASSES[args_dict["mimic_task"]]
    
    return args_dict, rng_key

    # END OF GENERAL SETUP FOR SCRIPT and MODEL ARCHITECTURE
    #-----------------------------------------------------------------------------