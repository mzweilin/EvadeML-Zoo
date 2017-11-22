from future.standard_library import install_aliases
install_aliases()
from urllib import parse as urlparse

def parse_params(params_str):
    if '?' not in params_str:
        return params_str, {}
    subject, params_str = params_str.split('?')
    params = urlparse.parse_qs(params_str)
    params = dict( (k, v if len(v)>1 else v[0] ) for k,v in params.items())
    return subject, params