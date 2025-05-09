from LBSNData import LBSNData
from near_location_query import Loc_Query_System
from utils import *
from loss_fn import *
from neg_sampler import *
from model import STiSAN
from trainer import *
import os
from pathlib import Path
from options import args


if __name__ == "__main__":
    # Setting paths
    set_seed(args.random_seed)
    current_file_path = os.path.abspath(__file__)
    current_dir_name = Path(current_file_path).parent.name

    data_name = args.data_name
    print("Dataset: ", data_name)

    # read file
    raw_data_prefix = args.raw_data_prefix
    raw_data_path = raw_data_prefix + data_name + '.txt'

    # mk_dir
    output_prefix = args.out_prefix
    file_prefix = output_prefix + current_dir_name + '/' + data_name + '/'
    temp_path_prefix = mk_dir(file_prefix + 'temp/')
    log_path_prefix = mk_dir(file_prefix + 'log/')
    result_path_prefix = mk_dir(file_prefix + 'result/')
    model_path_prefix = mk_dir(file_prefix + 'model/')

    # temp file
    if os.path.exists(raw_data_prefix + data_name + '_loc_query.pkl'):
        loc_query_path = raw_data_prefix + data_name + '_loc_query.pkl'
    else:
        loc_query_path = temp_path_prefix + data_name + '_loc_query.pkl'

    if os.path.exists(raw_data_prefix + data_name + '.data'):
        clean_data_path = raw_data_prefix + data_name + '.data'
    else:
        clean_data_path = temp_path_prefix + data_name + '.data'
    matrix_path = temp_path_prefix + data_name + '_st_matrix.data'

    # log file
    log_path = log_path_prefix + data_name + '_log.txt'
    # result file
    result_path = result_path_prefix + data_name + '_result.txt'
    # model file
    model_path = model_path_prefix

    # Data Process details
    min_loc_freq = args.min_loc_freq
    min_user_freq = args.min_user_freq
    map_level = args.map_level
    n_nearest = args.n_nearest
    max_len = args.max_len

    if os.path.exists(clean_data_path) and args.new_build:
        print("Old Dataset")
        dataset = unserialize(clean_data_path)
    else:
        print("New Dataset")
        dataset = LBSNData(data_name, raw_data_path,
                           min_loc_freq, min_user_freq, map_level)
        serialize(dataset, clean_data_path)
    count = 0
    length = []
    for seq in dataset.user_seq:
        count += len(seq)
        length.append(len(seq))
    print("#check-ins:", count)
    print("#users:", dataset.n_user - 1)
    print("#POIs:", dataset.n_loc - 1)
    print("#median seq len:", np.median(np.array(length)))

    # Searching nearest POIs
    quadkey_processor = dataset.GPSQUADKEY
    timestamp_processor = dataset.TIMECODE
    print("#quadkey:",len(quadkey_processor.vocab.itos))

    loc2quadkey = dataset.loc2quadkey
    n_user = dataset.n_user
    n_loc = dataset.n_loc
    n_timestamp = dataset.n_timestamp

    loc_query_sys = Loc_Query_System()
    if os.path.exists(loc_query_path) and args.new_build:
        loc_query_sys.load(loc_query_path)
    else:
        loc_query_sys.build_tree(dataset)
        loc_query_sys.prefetch_n_nearest_locs(n_nearest)
        loc_query_sys.save(loc_query_path)
        loc_query_sys.load(loc_query_path)

    # Building Spatial-Temporal Relation Matrix
    # if os.path.exists(matrix_path) and args.new_build:
    #     st_matrix = unserialize(matrix_path)
    # else:
    #     dataset.spatial_temporal_matrix_building(matrix_path)
    #     st_matrix = unserialize(matrix_path)
    # print("Data Partition...")
    # train_data, eval_data = dataset.data_partition(max_len, st_matrix)
    train_data, eval_data = dataset.data_partition(max_len)

    # Setting training details
    device = torch.device('cuda')

    num_workers = args.num_workers
    n_nearest_locs = args.n_nearest_locs
    num_epoch = args.n_epoch
    train_bsz = args.train_batch_size
    eval_bsz = args.eval_batch_size
    train_num_neg = args.train_n_neg
    eval_num_neg = args.eval_n_neg
    user_visited_locs = get_visited_locs(dataset)
    loss_fn = WeightedBCELoss(temperature=args.exp_factor)
    train_sampler = KNNSampler(
        loc_query_sys, n_nearest_locs, user_visited_locs, 'training', args.unvisit_loc_t)
    eval_sampler = KNNSampler(
        loc_query_sys, n_nearest_locs, user_visited_locs, 'training', args.unvisit_loc_e)

    # Model details
    model = STiSAN(n_user,
                   dataset.n_loc,
                   len(quadkey_processor.vocab.itos),
                   n_timestamp,
                   features=args.dimension,
                   exp_factor=args.exp_factor,
                   k_t=args.k_t,
                   k_d=args.k_d,
                   depth=args.depth,
                   src_len =max_len,
                   dropout=args.dropout,
                   device=device)

    if args.parallel:
        if torch.cuda.device_count() > 1:
            model.cuda()
            model = nn.parallel.DataParallel(
                model, device_ids=list(range(torch.cuda.device_count())))
    else:
        model.to(device)

    # Starting training

    # weight_decay = 0.1
    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001, betas=(0.9, 0.999))
    train(model,
          max_len,
          train_data,
          train_sampler,
          train_bsz,
          train_num_neg,
          num_epoch,
          quadkey_processor,
          timestamp_processor,
          loc2quadkey,
          eval_data,
          eval_sampler,
          eval_bsz,
          eval_num_neg,
          optimizer,
          loss_fn,
          device,
          num_workers,
          log_path,
          result_path,
          model_path,
          args.parallel)
