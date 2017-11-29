import sys, os

external_libs = {'Cleverhans': "externals/cleverhans",
                 "Carlini_nn_robust_attacks": "externals/carlini",
                 "Keras-deep-learning-models": "externals/keras_models",
                 "MobileNets": "externals/MobileNetworks",
                 "Deepfool/Universal": "externals/universal/python",
                 "DenseNet": "externals/titu1994/DenseNet",
                 "MagNet": "externals/MagNet",
                 }

project_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

for lib_name, lib_path in external_libs.items():
    lib_path = os.path.join(project_path, lib_path)

    if lib_name == 'Carlini_nn_robust_attacks':
        lib_token_fpath = os.path.join(lib_path, 'nn_robust_attacks', '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    if lib_name == 'Keras-deep-learning-models':
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    if lib_name == 'MobileNets':
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    if lib_name == 'Deepfool/Universal':
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    if lib_name == 'DenseNet':
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()

    if lib_name in ['MagNet']:
        lib_token_fpath = os.path.join(lib_path, '__init__.py')
        if not os.path.isfile(lib_token_fpath):
            open(lib_token_fpath, 'a').close()
    
    sys.path.append(lib_path)
    print("Located %s" % lib_name)
