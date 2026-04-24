import pandas as pd
import os
from pathlib import Path

# parquet 文件所在总目录
base_path = r"D:\c\LQ_20260422_fail_2_1776850638444_lerobot\trainable\data"

# 输出 CSV 的保存目录
output_path = r"D:\c\parquet_to_csv_output"

# 如果输出目录不存在，就自动创建
os.makedirs(output_path, exist_ok=True)

# 遍历所有子文件夹和 parquet 文件
for root, dirs, files in os.walk(base_path):
    for file in files:
        if file.endswith(".parquet"):
            parquet_file = os.path.join(root, file)

            try:
                print(f"正在转换：{parquet_file}")

                # 读取 parquet
                df = pd.read_parquet(parquet_file, engine="pyarrow")

                # 保持原文件名，只改后缀为 csv
                csv_name = Path(file).stem + ".csv"
                csv_file = os.path.join(output_path, csv_name)

                # 保存 csv
                df.to_csv(
                    csv_file,
                    index=False,
                    encoding="utf-8-sig"
                )

                print(f"已保存：{csv_file}")

            except Exception as e:
                print(f"转换失败：{parquet_file}")
                print(f"错误信息：{e}")

print("全部转换完成！")