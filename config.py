import yaml
from pprint import pprint


def join(loader, node):
    seq = loader.construct_sequence(node)
    return "".join([str(i) for i in seq])


yaml.add_constructor("!join", join)

with open("config.yaml", "r") as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)
    pprint("*** Configurations: ***")
    pprint(config)
    pprint("*** Config loaded succesfully. ***")
    pprint("***" * 7)
