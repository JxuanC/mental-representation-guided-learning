import os
import shutil
import torch
from PIL import Image
import logging.config
from datetime import datetime
import torch.utils.data
from torch.nn import functional as F
from einops import rearrange
from torchvision.utils import save_image

def save_checkpoint(model, epoch, batch, save_path, save_name = 'model'):
    os.makedirs(os.path.join(save_path, 'checkpoints'), exist_ok = True)
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoints', f'{save_name}_{epoch:0>3d}_{batch}.pth'))
    torch.save(model.state_dict(), os.path.join(save_path, 'checkpoints', 'last.pth'))

def save_images(save_path, save_name, images, normalize = False, nrow = None):
    save_dir = os.path.join(save_path, save_name + '.png')
    n = min(images[0].size(0), 20) if nrow is None else nrow
    comparison = torch.cat([images[i][:n] for i in range(len(images))])
    save_image(comparison.cpu(), save_dir, nrow = n, normalize = normalize)

def setup_logging_from_args(results_dir = './results', save_name = ''):
    """
    Calls setup_logging, exports args and creates a ResultsLog class.
    Can resume training/logging if args.resume is set
    """
    if save_name == '':
        save_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    save_path = os.path.join(results_dir, save_name)
    if os.path.exists(save_path):
        shutil.rmtree(save_path)
    os.makedirs(save_path, exist_ok = True)
    log_file = os.path.join(save_path, 'log.txt')
    setup_logging(log_file)
    return save_path

def setup_logging(log_file='log.txt', resume = False):
    """
    Setup logging configuration
    """
    if os.path.isfile(log_file) and resume:
        file_mode = 'a'
    else:
        file_mode = 'w'

    root_logger = logging.getLogger()
    if root_logger.handlers:
        root_logger.removeHandler(root_logger.handlers[0])
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S",
                        filename=log_file,
                        filemode=file_mode)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(message)s')
    console.setFormatter(formatter)
    logging.getLogger(datetime.now().strftime('%Y-%m-%d_%H-%M-%S')).addHandler(console)