import numpy as np
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass

from argparse import ArgumentParser
from glob import glob
from natsort import natsorted
from ml_networks import load_blosc2, save_blosc2
import os
import yaml

@dataclass
class IDList:
    """データセットのIDを管理するクラス"""
    train_ids: np.ndarray
    val_ids: np.ndarray
    all_ids: np.ndarray

def split_id(
    num: int,
    each_val_num: int,
    change_point: Optional[Union[Tuple[int], List[int]]] = None,
) -> IDList:
    """
    データセットのIDを訓練用と検証用に分割する関数

    Args:
        num (int): 全データ数
        each_val_num (int): 各区間から選択する検証データの数
        change_point (Optional[Union[int, List[int], tuple]]): 検証データの選択方法を指定
            - None: ランダムに選択
            - List[int] or tuple: 指定した区間ごとに検証データを選択

    Returns:
        IDList: 訓練用、検証用、全データのIDを含むオブジェクト

    Examples
    --------

    >>> ids = split_id(100, 5, [25, 50, 75])
    >>> ids.train_ids.shape
    (80,)
    >>> ids.val_ids.shape
    (20,)



    """
    all_ids = np.arange(num)
    
    if change_point is None:
        # ランダムに検証データを選択
        val_ids = np.random.choice(num, each_val_num, replace=False)
    else:
        # 区間ごとに検証データを選択
        change_points = np.array([0, *change_point, num])
        val_ids = []
        
        for i in range(len(change_points) - 1):
            start, end = change_points[i], change_points[i + 1]
            interval_ids = all_ids[start:end]
            selected_ids = np.random.choice(
                len(interval_ids),
                min(each_val_num, len(interval_ids)),
                replace=False
            )
            val_ids.extend(interval_ids[selected_ids])
        
        val_ids = np.array(val_ids)

    # 訓練データのIDを取得
    train_ids = np.delete(all_ids, val_ids)
    # 訓練データのIDをシャッフル
    train_ids = np.random.permutation(train_ids)

    print(f'Train: size {len(train_ids)} : {train_ids}')
    print(f'Validation: size {len(val_ids)} : {val_ids}')

    return IDList(train_ids=train_ids, val_ids=val_ids, all_ids=all_ids)


if __name__ == "__main__":
    # Example usage
    parser = ArgumentParser(description="Split dataset into train and validation sets.")
    parser.add_argument("--path", type=str, default="dataset", help="Path to the dataset.")
    parser.add_argument("--each-val-num", type=int, default=5, help="Number of validation samples per interval.")
    parser.add_argument("--change-point", type=int, nargs='+', default=None, help="Change points for validation samples.")
    parser.add_argument("--doctest", action="store_true", help="Run tests.", default=False)

    args = parser.parse_args()

    if args.doctest:
        import doctest
        doctest.testmod()

    else:
        # データセットのパスを取得
        dataset_path = args.path
        # データセットのファイルを取得
        dataset_files = natsorted(glob(f"{dataset_path}/*.blosc2"))

        num = len(load_blosc2(dataset_files[0]))

        # IDを分割
        id_list = split_id(num, args.each_val_num, args.change_point)

        cfg = {
            "train_ids": id_list.train_ids,
            "val_ids": id_list.val_ids,
        }
        os.makedirs(f"{dataset_path}/train", exist_ok=True)
        os.makedirs(f"{dataset_path}/val", exist_ok=True)

        for path in dataset_files:
            # データセットを読み込む
            data = load_blosc2(path)
            num_samples = len(data)
            assert num == num_samples, f"Data length mismatch: {num} != {num_samples}"

            data_name = path.split("/")[-1].split(".")[0]

            if data.ndim == 3:
                max = np.max(data, axis=[0, 1])
                min = np.min(data, axis=[0, 1])
                mean_init = np.mean(data[:, 0], axis=[0])
                cfg[f"{data_name}_max"] = max
                cfg[f"{data_name}_min"] = min
                cfg[f"{data_name}_mean_init"] = mean_init
            save_blosc2(
                f"{dataset_path}/train/{data_name}.blosc2",
                data[id_list.train_ids]
            )
            save_blosc2(
                f"{dataset_path}/val/{data_name}.blosc2",
                data[id_list.val_ids]
            )
        # 設定を保存
        with open(f"{dataset_path}/cfg.yaml", "w") as f:
            yaml.safe_dump(cfg, f)






