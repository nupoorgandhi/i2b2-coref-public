import os
from shutil import copyfile


def setup_dir(log_root, model_name):
    """
    create a new directory for model and copy over the checkpoint from original_dir_name
    :param dir_name:
    :return:
    """
    original_dir_name = os.path.join(log_root, model_name)
    dir_name = original_dir_name + '_tuneTrue'
    print('[setup_dir] original_dir_name', original_dir_name, 'new dir name', dir_name)
    default_checkpoint_file = '{}/checkpoint'.format(original_dir_name)
    print('default file', default_checkpoint_file)
    if os.path.exists(default_checkpoint_file):
        mkdirs(dir_name)
        if not os.path.exists(os.path.join(dir_name, 'checkpoint')):
            copyfile(default_checkpoint_file, os.path.join(dir_name, 'checkpoint'))
            return True
        else:
            print('checkpoint already exists:', os.path.join(dir_name, 'checkpoint'))
            return False
    else:
        return False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--log_root", type=str, help='High level coref model dir')
    parser.add_argument("--model_name", type=str, default="", help='Name of model to tune on target data')

    args = parser.parse_args()
    if setup_dir(args.log_root, args.model_name):
        print('successfully set up checkpoint')
    else:
        print('failed to set up checkpoint')
