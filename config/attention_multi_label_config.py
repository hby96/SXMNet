import os
import argparse


# Experiment root
exp_root = '../exp/'

# Experiment name
exp_name = 'xray'

# Experiment path
exp_path = os.path.join(exp_root, exp_name)

def create_parser():

    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    # Exp name
    parser.add_argument('--exp_root', default=exp_root)
    parser.add_argument('--exp_name', default=exp_name)
    parser.add_argument('--exp_path', default=exp_path)

    # Model save path
    parser.add_argument('--model_save_path', default=None)

    # Resume path
    parser.add_argument('--resume_path', default=None)

    # Log save path
    parser.add_argument('--log_save_path', default=os.path.join(exp_path, 'logs'))

    # Dataset Location
    parser.add_argument('--train_set_path', default='/home/hby/Documents/Datasets/Top_View_Xray/train_test_split/train_w_hby_wo_knife/', type=str,
                        help='root path of the train set')
    # parser.add_argument('--test_set_path', default='../data_set/final_divide/data_train_test_split/test/', type=str,
    #                     help='root path of the test path')
    parser.add_argument('--test_set_path', default='/home/hby/Documents/Datasets/Top_View_Xray/total_test_imgs/', type=str,
                        help='root path of the test path')
    parser.add_argument('--mask_path', default='../data_set/final_dataset/train_test_split/train/Heatmaps/', type=str,
                        help='root path of the test path')
    parser.add_argument('--classes',
                        default=['battery', 'bottle', 'firecracker', 'grenade', 'gun', 'hammer', 'knife', 'scissors'])
    parser.add_argument('--image_size', default=224, type=int, help='batch size of the image')

    # Label nums
    parser.add_argument('--num_classes', type=int, default=8, help='num classes of the dataset')

    # Optimization parameters
    parser.add_argument('--epochs', '-e', type=int, default=155, help='Number of epochs to train.')#95
    parser.add_argument('--lr-rampup', default=5, type=int, metavar='EPOCHS', help='length of learning rate rampup in the beginning')
    parser.add_argument('--lr-rampdown-epochs', default=180, type=int, metavar='EPOCHS',
                            help='length of learning rate cosine rampdown (>= length of training)')
    parser.add_argument('--batch_size', '-b', type=int, default=64, help='Batch size.')#64
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-2, help='The Learning Rate.')#双池化输出: 5e-3 gap: 5e-3 gmp:1e-3 gap gmp concat:1e-3 best_employ:1e-2
    parser.add_argument('--initial-lr', default=0.0, type=float,
                            metavar='LR', help='initial learning rate when using linear rampup')
    parser.add_argument('--momentum', '-m', type=float, default=0.9, help='Momentum.')
    parser.add_argument('--decay', '-d', type=float, default=1e-4, help='Weight decay (L2 penalty).')
    parser.add_argument('--schedule', type=int, default=30, help='Decrease learning rate at these epochs.')#30
    parser.add_argument('--gamma', type=float, default=0.1, help='LR is multiplied by gamma on schedule.')

    # Acceleration
    parser.add_argument('--n_gpu', type=int, default=2, help='0 = CPU.')

    return parser


def parse_commandline_args():
    return create_parser().parse_args()

