"""
Step 1: Please follow the Xgen user manual to prepare your training script.
Step 2: Put this script into the directory that contains the training script train_script_main.py
Step 3: Run XGen.

Example : Python xgen.py --xgen-config-path=xxxx/xxx/xx.json --xgen-mode='compitable_test' --xgen-pretrained-model-path=xxx/xxx/xxx.pth
"""
import datetime
import os
import sys
import traceback
import loguru

from xgen.training import training
from xgen.log.logger import logger, xgen_logger_setup
sys.path.append('/usr/local/compiler/cocogen/')  # Compiler in Docker
from cocogen import run


# def run_mock(onnx_path, quantized, pruning, output_path, **kwargs):
#     import random
#     res = {}
#     # for simulation
#     pr = kwargs['sp_prune_ratios']
#     num_blocks = kwargs.get('num_blocks', None)
#     res['output_dir'] = output_path
#     try:
#         if quantized:
#             res['latency'] = 50
#         else:
#             if num_blocks is not None:
#                 res['latency'] = 10 * (num_blocks) * (num_blocks)
#             else:
#                 res['latency'] = 100 - (pr * 10) * (pr * 10) - random.uniform(0, 10)
#     except:
#         res['latency'] = 50
#     return res

def inverse_clever_format(cfstr):
    num = cfstr[:-1]
    C = cfstr[-1]
    num = float(num)
    if C == "T":
        num = num * 1e12
    elif C == "G":
        num = num * 1e9
    elif C == "M":
        num = num * 1e6
    elif C == "K":
        num = num * 1e3
    return num


class CompilerSimulator:
    def __init__(self, base=None):
        self.base = base

    def __call__(self, onnx_path, quantized, pruning, output_path, **kwargs):

        res = {}
        # for simulation
        res['latency'] = 50
        if "internal_data" in kwargs:
            total_nz_params = kwargs['internal_data']['total_nz_parameters']
            total_nz_params = inverse_clever_format(total_nz_params)
            if self.base is None:
                self.base = total_nz_params
            res['latency'] = 100 * total_nz_params / self.base
        res['output_dir'] = output_path

        return res


run_mock = CompilerSimulator()


# make Logger a file-like object
# should inherit from one of io.FileIO , io.IOBase, io.TextIOBase
# this is just a temporary solution
class NotUseLogger(object):

    def __init__(self, filename="log.txt", *args, **kwargs):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        # do not close stdout
        self.log.close()

    # def __getattr__(self, item):
    #     """
    #     when visit an non-existed attribute, return `object` instead
    #     """
    #     warnings.warn(f'Non-existed attribute {str(item)} is visited on logger.')
    #     return object


def parse_workplace(args: list) -> str:
    for argv in args:
        kwarg = argv.split('=')
        if len(kwarg) != 2:
            continue
        key, value = kwarg
        if key == '--xgen-workplace':
            return value
    return os.getcwd()


def parse_bypass(args: list) -> bool:
    for argv in args:
        kwarg = argv.split('=')
        if len(kwarg) != 2:
            continue
        key, value = kwarg
        if key == '--xgen-bypass':
            return True if value == 'True' else False
    return False

def parse_verbose(args: list) -> bool:
    for argv in args:
        kwarg = argv.split('=')
        if len(kwarg) != 2:
            continue
        key, value = kwarg
        if key == '--xgen-verbose':
            return True if value == 'True' else False
    return False


@loguru.logger.catch
def main():
    workplace = parse_workplace(sys.argv)
    if not os.path.exists(workplace):
        os.makedirs(workplace, exist_ok=True)
    # now = datetime.datetime.now()
    # logfile_name = f"info_{now.strftime('%Y%m%d%H%M%S')}.log"
    logfile_name = 'xgen-training.log'
    verbose = parse_verbose(sys.argv)
    xgen_logger_setup(workplace, logfile_name, verbose)
    logger.debug(f"XGen is running in {workplace}, xgen_logger_setup in xgen_scripts.py")
    # FIXME: this is a old solution, need to be fixed
    log_path = os.path.join(workplace, logfile_name)
    # sys.stdout = Logger(log_path)
    # sys.stderr  = Logger(log_path)
    try:
        sys_args_back = sys.argv
        sys.argv = [sys.argv[0]]
        from train_script_main import training_main

        sys.argv = sys_args_back
        training_script_path = os.path.dirname(__file__)
        bypass = parse_bypass(sys.argv)
        if bypass:
            training(training_main, run_mock, training_script_path=training_script_path, log_path=log_path)
        else:
            training(training_main, run, training_script_path=training_script_path, log_path=log_path)
    except Exception:
        print(traceback.format_exc())
        with open(log_path, 'a') as f:
            f.write(traceback.format_exc())
        logger.error(f'Error found. Please check log file at {log_path}')


if __name__ == '__main__':
    main()
