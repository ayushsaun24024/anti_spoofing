import os
import yaml
import argparse

def read_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    args = parser.parse_args()
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    return config, args.exp_name


class Logs:
    def __init__(self, basepath, exp_name):
        self.chkpt_folder = os.path.join(basepath, "RESULTS", exp_name.upper())
        if not os.path.isdir(self.chkpt_folder): os.makedirs(self.chkpt_folder)
        with open(os.path.join(self.chkpt_folder, "LOG_FILE.txt"), "w") as F:
            F.write(
                "{} Experiment Results\n".format(exp_name.upper())
            )
            F.write("\n")
        F.close()
    def write(self, *args, **kwargs):
        data = " ".join([str(i) for i in args])
        with open(os.path.join(self.chkpt_folder, "LOG_FILE.txt"), "a") as F:
            F.write(">>  "+data+"\n")
        F.close()