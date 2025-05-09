from batch_generater import cf_eval_quadkey
from utils import *
from torch.utils.data import DataLoader
from collections import Counter


def collate_eval_batch(batch):
    return cf_eval_quadkey(
        batch,
        eval_data=batch,  # This will be overwritten in evaluate function
        max_len=100,  # This will be overwritten in evaluate function
        sampler=None,  # This will be overwritten in evaluate function
        quadkey_processor=None,  # This will be overwritten in evaluate function
        timestamp_processor=None,  # This will be overwritten in evaluate function
        loc2quadkey=None,  # This will be overwritten in evaluate function
        eval_num_neg=100  # This will be overwritten in evaluate function
    )


def evaluate(model, max_len, eval_data, eval_sampler, eval_batch_size, eval_num_neg, quadkey_processor,
             timestamp_processor, loc2quadkey, device, num_workers):
    model.eval()

    # Create a modified collate function with the correct parameters
    def collate_fn(batch):
        return cf_eval_quadkey(
            batch,
            eval_data,
            max_len,
            eval_sampler,
            quadkey_processor,
            timestamp_processor,
            loc2quadkey,
            eval_num_neg
        )

    # Set num_workers=0 for Windows compatibility
    loader = DataLoader(
        eval_data,
        batch_size=eval_batch_size,
        num_workers=0,  # Set to 0 for Windows compatibility
        collate_fn=collate_fn
    )

    cnt = Counter()
    array = np.zeros(1 + eval_num_neg)

    with torch.no_grad():
        for _, (src_user_, src_locs_, src_quadkeys_, src_times_, src_timecode_, src_lat_, src_lng_,
                trg_locs_, trg_quadkeys_, trg_times_, trg_time_grams_, data_size) in enumerate(loader):
            src_user = src_user_.to(device)
            src_loc = src_locs_.to(device)
            src_quadkey = src_quadkeys_.to(device)
            src_timecode = src_timecode_.to(device)
            src_time = src_times_.to(device)
            src_lat = src_lat_.to(device)
            src_lng = src_lng_.to(device)
            trg_loc = trg_locs_.to(device)
            trg_times = trg_times_.to(device)
            trg_quadkey = trg_quadkeys_.to(device)
            trg_time_grams = trg_time_grams_.to(device)

            pad_mask = get_pad_mask(data_size, max_len, device)
            attn_mask = get_attn_mask(max_len, device)
            mem_mask = None
            key_pad_mask = None

            output = model(
                src_user, src_loc, src_quadkey, src_time, src_timecode,
                src_lat, src_lng, pad_mask, attn_mask,
                trg_loc, trg_quadkey, trg_times, trg_time_grams,
                key_pad_mask, mem_mask, data_size, False
            )

            idx = output.sort(descending=True, dim=1)[1]
            order = idx.topk(k=1, dim=1, largest=False)[1]
            cnt.update(order.squeeze(-1).tolist())

    for k, v in cnt.items():
        array[k] = v

    Hit_Rate = array.cumsum()
    NDCG = 1 / np.log2(np.arange(0, eval_num_neg + 1) + 2)
    NDCG = NDCG * array
    NDCG = NDCG.cumsum() / Hit_Rate.max()
    Hit_Rate = Hit_Rate / Hit_Rate.max()

    return Hit_Rate, NDCG