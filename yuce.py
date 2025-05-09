import torch
import os
from datetime import datetime
from model import STiSAN
from utils import get_pad_mask, get_attn_mask, haversine, unserialize
from quadkey_encoder import latlng2quadkey
from nltk import ngrams
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class LocationPredictor:
    def __init__(self, cache_dir, model_path, device='cuda'):
        """
        初始化位置预测器

        Args:
            cache_dir (str): 缓存文件目录,包含test.data和test_loc_query.pkl
            model_path (str): 模型文件路径
            device (str): 使用的设备 ('cuda' 或 'cpu')
        """
        self.device = device
        logger.info("正在初始化LocationPredictor...")

        # 加载缓存的数据集
        cache_data_path = os.path.join(cache_dir, 'testb.data')
        if not os.path.exists(cache_data_path):
            raise FileNotFoundError(f"找不到缓存数据文件: {cache_data_path}")

        logger.info("正在加载缓存的数据集...")
        self.dataset = unserialize(cache_data_path)
        logger.info(f"数据集加载完成，包含 {self.dataset.n_loc - 1} 个位置和 {self.dataset.n_user - 1} 个用户")

        # 加载模型
        self.setup_model(model_path)

    def setup_model(self, model_path):
        """初始化并加载模型"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"找不到模型文件: {model_path}")

        logger.info("正在加载模型...")
        self.model = STiSAN(
            n_user=self.dataset.n_user,
            n_loc=self.dataset.n_loc,
            n_quadkey=len(self.dataset.GPSQUADKEY.vocab.itos),
            n_timestamp=self.dataset.n_timestamp,
            features=50,
            exp_factor=1,
            k_t=10,
            k_d=15,
            depth=4,
            src_len=100,
            dropout=0.7,
            device=self.device
        )

        try:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()
            logger.info("模型加载成功")
        except Exception as e:
            logger.error(f"模型加载失败: {str(e)}")
            raise

    def prepare_input_tensor(self, user_idx, loc_idx, lat, lng, timestamp, time_feat, quadkey_tokens):
        """准备模型输入张量"""
        batch_size = 1
        seq_len = 100

        # 源序列张量
        src_user = torch.tensor([[user_idx]]).repeat(1, seq_len).to(self.device)
        src_loc = torch.tensor([[loc_idx]]).repeat(1, seq_len).to(self.device)
        src_time = torch.tensor([[timestamp]]).float().repeat(1, seq_len).to(self.device)
        src_timecode = torch.tensor([time_feat]).float().repeat(1, seq_len).view(1, seq_len, 5).to(self.device)
        src_lat = torch.tensor([[float(lat)]]).float().repeat(1, seq_len).to(self.device)
        src_lng = torch.tensor([[float(lng)]]).float().repeat(1, seq_len).to(self.device)

        # Quadkey处理
        numeric_quadkey = self.dataset.GPSQUADKEY.numericalize(quadkey_tokens)
        src_quadkey = numeric_quadkey.unsqueeze(0).repeat(1, seq_len, 1).to(self.device)

        # 目标序列张量
        trg_loc = torch.zeros(batch_size, seq_len).long().to(self.device)
        trg_quadkey = torch.zeros(batch_size, seq_len, src_quadkey.size(-1)).long().to(self.device)
        trg_times = torch.zeros(batch_size, seq_len).float().to(self.device)
        trg_time_grams = torch.zeros(batch_size, seq_len, 5).float().to(self.device)

        # 掩码
        pad_mask = get_pad_mask([seq_len], seq_len, self.device)
        attn_mask = get_attn_mask(seq_len, self.device)

        return {
            'src': (src_user, src_loc, src_quadkey, src_time, src_timecode, src_lat, src_lng),
            'trg': (trg_loc, trg_quadkey, trg_times, trg_time_grams),
            'mask': (pad_mask, attn_mask)
        }

    def predict_next_location(self, user_id, current_loc_id, lat, lng, timestamp):
        """预测下一个可能的位置"""
        try:
            logger.info(f"开始预测: user_id={user_id}, loc_id={current_loc_id}, coordinates=({lat}, {lng})")

            # 检查并获取位置索引
            if str(current_loc_id) not in self.dataset.loc2idx:
                raise ValueError(f"位置ID {current_loc_id} 在数据集中不存在")
            loc_idx = self.dataset.loc2idx[str(current_loc_id)]

            # 处理用户ID
            user_idx = 1  # 默认用户ID，因为在预测时用户ID不影响结果

            # 时间处理
            dt = datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%SZ")
            timestamp_secs = dt.timestamp()
            time_feat = [dt.month, dt.day, dt.weekday(), dt.hour, dt.minute // 15]

            # Quadkey处理
            quadkey = latlng2quadkey(float(lat), float(lng), 17)
            quadkey_bigrams = []
            for bg in ngrams(quadkey, 6):
                bigram = ''.join(bg)
                quadkey_bigrams.append(bigram)

            # 准备输入数据
            tensors = self.prepare_input_tensor(
                user_idx, loc_idx, lat, lng, timestamp_secs,
                time_feat, quadkey_bigrams
            )

            # 模型预测
            with torch.no_grad():
                output = self.model(
                    *tensors['src'],
                    tensors['mask'][0],
                    tensors['mask'][1],
                    *tensors['trg'],
                    None, None, [1], False
                )

                scores, indices = torch.topk(output, k=10, dim=1)

                # 处理预测结果
                predictions = []
                for score, idx in zip(scores[0].cpu().numpy(), indices[0].cpu().numpy()):
                    pred_loc_id = self.dataset.idx2loc[idx]
                    pred_lat, pred_lng = self.dataset.idx2gps[idx]
                    distance = haversine((float(lat), float(lng)), (pred_lat, pred_lng))
                    predictions.append({
                        'location_id': pred_loc_id,
                        'score': float(score),
                        'latitude': pred_lat,
                        'longitude': pred_lng,
                        'distance_km': distance
                    })

                logger.info(f"成功生成 {len(predictions)} 个预测结果")
                return predictions

        except Exception as e:
            logger.error(f"预测过程发生错误: {str(e)}")
            raise


def main():
    """主函数"""
    try:
        # 配置路径
        cache_dir = "output/shikongjuli/testb/temp"
        model_path = "output/shikongjuli/testb/model/model_epoch10_H5_0.15_N5_0.11_H10_0.22_N10_0.13.pth"

        print("=== 位置预测系统 ===")
        predictor = LocationPredictor(cache_dir, model_path)

        # 示例数据
        sample_data = [
            "0\t2010-10-19T23:55:27Z\t30.2359091167\t-97.7951395833\t22847",
            "0\t2010-10-18T22:17:43Z\t30.2691029532\t-97.7493953705\t420315",
            "0\t2010-10-17T23:42:03Z\t30.2557309927\t-97.7633857727\t316637",
            "0\t2010-10-17T19:26:05Z\t30.2634181234\t-97.7575966669\t16516",
            "0\t2010-10-16T18:50:42Z\t30.2742918584\t-97.7405226231\t5535878",
            "0\t2010-10-12T23:58:03Z\t30.261599404\t-97.7585805953\t15372",
            "0\t2010-10-12T22:02:11Z\t30.2679095833\t-97.7493124167\t21714",
            "0\t2010-10-12T19:44:40Z\t30.2691029532\t-97.7493953705\t420315",
            "0\t2010-10-12T15:57:20Z\t30.2811204101\t-97.7452111244\t153505",
            "0\t2010-10-12T15:19:03Z\t30.2691029532\t-97.7493953705\t420315",
            "0\t2010-10-12T00:21:28Z\t40.6438845363\t-73.7828063965\t23261",
            "0\t2010-10-11T20:21:20Z\t40.74137425\t-73.9881052167\t16907"
        ]

        print("\n开始处理示例数据...")

        for line in sample_data:
            user_id, timestamp, lat, lng, loc_id = line.strip().split('\t')

            print(f"\n处理记录:")
            print(f"用户ID: {user_id}")
            print(f"时间: {timestamp}")
            print(f"位置: ({lat}, {lng})")
            print(f"位置ID: {loc_id}")

            try:
                predictions = predictor.predict_next_location(
                    int(user_id), loc_id, float(lat), float(lng), timestamp
                )

                print("\n预测的下一个可能位置:")
                print("排名  位置ID         得分        坐标                距离(km)")
                print("-" * 70)
                for i, pred in enumerate(predictions, 1):
                    print(f"{i:2d}    {pred['location_id']:<12} {pred['score']:6.4f}  "
                          f"({pred['latitude']:9.4f}, {pred['longitude']:9.4f})  {pred['distance_km']:8.2f}")

            except Exception as e:
                print(f"预测失败: {str(e)}")
                continue

            print("\n" + "=" * 70)

    except Exception as e:
        print(f"系统运行出错: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()