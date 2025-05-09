import argparse

parser = argparse.ArgumentParser()

# Environiment set
parser.add_argument('--random_seed', type=int,
                    default=42, help='golbal seed')
parser.add_argument('--raw_data_prefix', type=str,
                    default='../../data-1/data5k/')
parser.add_argument('--data_name', type=str, default='gowalla_5k')
parser.add_argument('--out_prefix', type=str, default='../../output/')
parser.add_argument('--parallel', type=bool, default=False)
parser.add_argument('--num_workers', type=int, default=24,
                    help='how many subprocesses to use for data loading')

# Data Process details
parser.add_argument('--new_build', type=bool, default=True,
                    help='whether use old LSBNData')
parser.add_argument('--min_loc_freq', type=int, default=0,
                    help='filter loc less than...')
parser.add_argument('--min_user_freq', type=int, default=0,
                    help='filter user less than ...')
parser.add_argument('--map_level', type=int, default=17,
                    help='quadkey encode map level')
parser.add_argument('--n_nearest', type=int,
                    default=2000, help='build ball tree')
parser.add_argument('--n_nearest_locs', type=int,
                    default=500, help='knn sample set')
parser.add_argument('--k_t', type=int, default=10, help='max time interval')
parser.add_argument('--k_d', type=int, default=15, help='max poi interval')

# Model set
parser.add_argument('--dimension', type=int, default=50,
                    help='embedding dimension')
parser.add_argument('--exp_factor', type=int, default=1, help='expand for FFN')
parser.add_argument('--depth', type=int, default=4, help='depth of net')
parser.add_argument('--max_len', type=int, default=100, help='max seq length')
parser.add_argument('--trg_len', type=int, default=1,
                    help='predict one or seq')
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--train_batch_size', type=int, default=8)
parser.add_argument('--eval_batch_size', type=int, default=8)
parser.add_argument('--n_epoch', type=int, default=0)

# Train and  Eval Set
parser.add_argument('--unvisit_loc_t', type=bool, default=False,
                    help='use unvisited loc when training')
# better True
parser.add_argument('--unvisit_loc_e', type=bool, default=False,
                    help='use unvisited loc when evaling')
parser.add_argument('--train_n_neg', type=int, default=15,
                    help='num of neg sample for train')
parser.add_argument('--eval_n_neg', type=int, default=500,
                    help='num of neg sample for eval')


args = parser.parse_args()
