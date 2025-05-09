import torch
from utils import fix_length
from einops import rearrange
import logging

# 设置日志
logging.basicConfig(level=logging.WARNING)  # 或者 logging.ERROR
logger = logging.getLogger(__name__)


def cf_train_quadkey(batch, data_source, max_len, sampler, quadkey_processor, TIME_processor, loc2quadkey, num_neg):
    src_seq, trg_seq = zip(*batch)
    src_user_, src_locs_, src_quadkeys_, src_timecode_, src_lat_, src_lng_, src_times_ = [], [], [], [], [], [], []
    data_size = []

    #logger.debug(f"Batch size: {len(batch)}")
    #logger.debug(f"Source sequence length: {len(src_seq)}")
    #logger.debug(f"Target sequence length: {len(trg_seq)}")

    valid_sequences = []  # 存储成功处理的序列

    for i, e in enumerate(src_seq):
        if not e:
            #logger.warning(f"Empty source sequence at index {i}")
            continue

        try:
            #logger.debug(f"Processing sequence {i}, length: {len(e)}")
            #logger.debug(f"First element in sequence: {e[0]}")

            u_, l_, q_, t_, lat_, lng_, tg_, _ = zip(*e)

            #logger.debug(f"Extracted values for sequence {i}:")
            #logger.debug(f"User IDs: {u_}")
            #logger.debug(f"Location IDs: {l_}")
            #logger.debug(f"Time values: {t_}")
            #logger.debug(f"Quadkeys: {q_}")  # 添加quadkey调试信息

            # 先创建临时变量验证quadkey处理
            try:
                temp_q = quadkey_processor.numericalize(list(q_))
                # 如果quadkey处理成功，才添加所有数据
                src_user_.append(torch.tensor(u_))
                src_lat_.append(torch.tensor(lat_))
                src_lng_.append(torch.tensor(lng_))
                data_size.append(len(u_))
                src_locs_.append(torch.tensor(l_))
                src_quadkeys_.append(temp_q)
                src_timecode_.append(torch.tensor(tg_))
                src_times_.append(torch.tensor(t_))
                valid_sequences.append(e)
            except Exception as ex:
                logger.error(f"Error processing quadkey at index {i}: {str(ex)}")
                continue

        except Exception as ex:
            logger.error(f"Error processing sequence at index {i}: {str(ex)}")
            logger.error(f"Sequence content: {e}")
            continue

    if not valid_sequences:
        logger.error("No valid sequences found in batch")
        raise ValueError("All sequences in batch are empty")

    logger.debug(f"Successfully processed {len(valid_sequences)} sequences")
    logger.debug(f"Quadkeys length: {len(src_quadkeys_)}")

    # 确保所有列表长度一致
    assert len(src_user_) == len(src_quadkeys_), "Inconsistent sequence lengths"

    src_user_ = fix_length(src_user_, 1, max_len, 'train src seq')
    src_lat_ = fix_length(src_lat_, 1, max_len, 'train src seq')
    src_lng_ = fix_length(src_lng_, 1, max_len, 'train src seq')
    src_locs_ = fix_length(src_locs_, 1, max_len, 'train src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'train src seq')
    src_timecode_ = fix_length(src_timecode_, 2, max_len, 'train src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'train src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    trg_time_grams_ = []
    trg_times_ = []

    for i, seq in enumerate(trg_seq):
        if not seq:
            logger.warning(f"Empty target sequence at index {i}")
            continue

        try:
            logger.debug(f"Processing target sequence {i}, content: {seq}")

            pos = torch.tensor([[e[1]] for e in seq])
            pos_time_grams = torch.tensor([[e[6]] for e in seq])
            trg_times = torch.tensor([[e[3]] for e in seq])
            neg = sampler(seq, num_neg, user=seq[0][0])
            pos_neg_locs = torch.cat([pos, neg], dim=-1)

            trg_times_.append(trg_times)
            trg_locs_.append(pos_neg_locs)
            trg_time_grams_.append(pos_time_grams)

            pos_neg_quadkey = []
            for l in range(pos_neg_locs.size(0)):
                q_key = []
                for loc_idx in pos_neg_locs[l]:
                    q_key.append(loc2quadkey[loc_idx])
                pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
            trg_quadkeys_.append(torch.stack(pos_neg_quadkey))

        except Exception as ex:
            logger.error(f"Error processing target at index {i}: {str(ex)}")
            continue

    if not trg_locs_:
        logger.error("No valid target sequences found in batch")
        raise ValueError("All target sequences in batch are empty")

    trg_locs_ = fix_length(trg_locs_, n_axies=2, max_len=max_len, dtype='train trg seq')
    trg_times_ = fix_length(trg_times_, n_axies=2, max_len=max_len, dtype='train trg seq')
    trg_times_ = rearrange(rearrange(trg_times_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')

    trg_time_grams_ = fix_length(trg_time_grams_, n_axies=3, max_len=max_len, dtype='train trg seq')
    trg_time_grams_ = rearrange(rearrange(trg_time_grams_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    trg_locs_ = rearrange(rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')

    trg_quadkeys_ = fix_length(trg_quadkeys_, n_axies=3, max_len=max_len, dtype='train trg seq')
    trg_quadkeys_ = rearrange(rearrange(trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_user_, src_locs_, src_quadkeys_, src_times_, src_timecode_, src_lat_, src_lng_, trg_locs_, trg_quadkeys_, trg_times_, trg_time_grams_, data_size

def cf_eval_quadkey(batch, data_source, max_len, sampler, quadkey_processor, timestamp_processor, loc2quadkey, num_neg):
    src_seq, trg_seq = zip(*batch)
    src_user_, src_locs_, src_quadkeys_, src_timecode_, src_lat_, src_lng_, src_times_ = [], [], [], [], [], [], []
    data_size = []

    for e in src_seq:
        # Skip if sequence is empty
        if not e:
            continue
        u_, l_, q_, t_, lat_, lng_, tg_, _ = zip(*e)
        src_user_.append(torch.tensor(u_))
        data_size.append(len(u_))
        src_locs_.append(torch.tensor(l_))
        src_lat_.append(torch.tensor(lat_))
        src_lng_.append(torch.tensor(lng_))
        q_ = quadkey_processor.numericalize(list(q_))
        src_quadkeys_.append(q_)
        src_timecode_.append(torch.tensor(tg_))
        src_times_.append(torch.tensor(t_))

    # Handle case when all sequences are empty
    if not src_user_:
        raise ValueError("All sequences in batch are empty")

    src_user_ = fix_length(src_user_, 1, max_len, 'eval src seq')
    src_lat_ = fix_length(src_lat_, 1, max_len, 'eval src seq')
    src_lng_ = fix_length(src_lng_, 1, max_len, 'eval src seq')
    src_locs_ = fix_length(src_locs_, 1, max_len, 'eval src seq')
    src_quadkeys_ = fix_length(src_quadkeys_, 2, max_len, 'eval src seq')
    src_timecode_ = fix_length(src_timecode_, 2, max_len, 'eval src seq')
    src_times_ = fix_length(src_times_, 1, max_len, 'eval src seq')

    trg_locs_ = []
    trg_quadkeys_ = []
    trg_time_grams_ = []
    trg_times_ = []

    for i, seq in enumerate(trg_seq):
        # Skip if sequence is empty
        if not seq:
            continue

        pos = torch.tensor([[e[1]] for e in seq])
        pos_time_grams = torch.tensor([[e[6]] for e in seq])
        trg_times = torch.tensor([[e[3]] for e in seq])
        neg_sample_from = [src_seq[i][-1]]
        neg = sampler(neg_sample_from, num_neg, user=neg_sample_from[0][0])
        pos_neg_locs = torch.cat([pos, neg], dim=-1)

        trg_times_.append(trg_times)
        trg_locs_.append(pos_neg_locs)
        trg_time_grams_.append(pos_time_grams)
        pos_neg_quadkey = []
        for l in range(pos_neg_locs.size(0)):
            q_key = []
            for loc_idx in pos_neg_locs[l]:
                q_key.append(loc2quadkey[loc_idx])
            pos_neg_quadkey.append(quadkey_processor.numericalize(q_key))
        trg_quadkeys_.append(torch.stack(pos_neg_quadkey))

    # Handle case when all target sequences are empty
    if not trg_locs_:
        raise ValueError("All target sequences in batch are empty")

    trg_locs_ = fix_length(trg_locs_, n_axies=2, max_len=max_len, dtype='eval trg loc')
    trg_times_ = fix_length(trg_times_, n_axies=2, max_len=max_len, dtype='eval trg loc')
    trg_times_ = rearrange(rearrange(trg_times_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')

    trg_time_grams_ = fix_length(trg_time_grams_, n_axies=3, max_len=max_len, dtype='eval trg loc')
    trg_time_grams_ = rearrange(rearrange(trg_time_grams_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    trg_locs_ = rearrange(rearrange(trg_locs_, 'b n k -> k n b').contiguous(), 'k n b -> b (k n)')
    trg_quadkeys_ = fix_length(trg_quadkeys_, n_axies=3, max_len=max_len, dtype='eval trg loc')
    trg_quadkeys_ = rearrange(rearrange(trg_quadkeys_, 'b n k l -> k n b l').contiguous(), 'k n b l -> b (k n) l')

    return src_user_, src_locs_, src_quadkeys_, src_times_, src_timecode_, src_lat_, src_lng_, trg_locs_, trg_quadkeys_, trg_times_, trg_time_grams_, data_size