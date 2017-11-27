from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

def isfloat(value):
    try:
        float(value)
        return True
    except:
        return False


def parse_params(params_str):
    if '?' not in params_str:
        return params_str, {}
    subject, params_str = params_str.split('?')
    params = urlparse.parse_qs(params_str)
    params = dict( (k, v.lower() if len(v)>1 else v[0] ) for k,v in params.items())

    # Data type conversion.
    integer_parameter_names = ['batch_size', 'max_iterations', 'num_classes', 'max_iter', 'nb_iter', 'max_iter_df']
    for k,v in params.items():
        if k in integer_parameter_names:
            params[k] = int(v)
        elif v == 'true':
            params[k] = True
        elif v == 'false':
            params[k] = False
        elif v == 'inf':
            params[k] = np.inf
        elif isfloat(v):
            params[k] = float(v)

    return subject, params