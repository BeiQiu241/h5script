import pandas as pd
import os
from pathlib import Path

# parquet 文件所在目录
base_path = r"D:\c\LQ_20260422_fail_2_1776850638444_lerobot\trainable\data"

# 输出 h5 文件保存目录
output_path = r"D:\c\parquet_to_h5_output"

# 自动创建输出目录
os.makedirs(output_path, exist_ok=True)

# 遍历所有 parquet 文件
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".parquet"):
            parquet_file = os.path.join(root, file)

            try:
                print(f"正在转换：{parquet_file}")

                # 读取 parquet 文件
                df = pd.read_parquet(
                    parquet_file,
                    engine="pyarrow"   # 如果你装的是 fastparquet，可改成 fastparquet
                )

                # 生成对应的 .h5 文件名
                h5_name = Path(file).stem + ".h5"
                h5_file = os.path.join(output_path, h5_name)

                # 保存为 HDF5 文件
                df.to_hdf(
                    h5_file,
                    key="data",     # HDF5 内部数据集名称
                    mode="w"
                )

                print(f"已保存：{h5_file}")

            except Exception as e:
                print(f"转换失败：{parquet_file}")
                print(f"错误信息：{e}")

print("全部转换完成！")