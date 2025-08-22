from datasets import load_dataset

def convert(sample):
    conversations = sample["conversations"]

    def convert_role(role):
        if role == "human":
            return "user"
        elif role == "gpt":
            return "assistant"
        elif role == "system":
            return "system"
        else:
            raise ValueError(f"Unknown role: {role}")

    messages = [
        {
            "role": convert_role(turn["from"]),
            "content": turn["value"],
        }
        for turn in conversations
    ]

    return {"messages": messages}

def convert_to_parquet_streaming(dataset, output_dir, batch_size=10000, max_rows=None):
    """
    流式处理大数据集并保存为多个 Parquet 文件
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    batch_count = 0
    total_rows = 0
    current_batch = []

    for example in tqdm(dataset, desc="Processing"):
        current_batch.append(example)

        if len(current_batch) >= batch_size:
            # 保存当前批次
            save_batch_as_parquet(current_batch, output_dir, batch_count)
            batch_count += 1
            total_rows += len(current_batch)
            current_batch = []

            # 可选：限制总行数
            if max_rows and total_rows >= max_rows:
                break

    # 保存最后一批
    if current_batch:
        save_batch_as_parquet(current_batch, output_dir, batch_count)


def save_batch_as_parquet(batch, output_dir, batch_id):
    """保存单个批次为 Parquet 文件"""
    import pyarrow as pa
    import pyarrow.parquet as pq
    import os

    # 转换为 PyArrow Table
    table = pa.Table.from_pylist(batch)

    # 保存文件
    output_path = os.path.join(output_dir, f"batch_{batch_id:06d}.parquet")
    pq.write_table(table, output_path, compression='snappy')

    print(f"Saved {len(batch)} rows to {output_path}")

ds = load_dataset("/root/wttest/data/sft/", streaming=True)["train"]

# 使用示例
convert_to_parquet_streaming(dataset,
                             output_dir='./parquet_data',
                             batch_size=5000)

ds = ds.map(convert)
ds.to_parquet("/root/wttest/data/sft/parquet/0726--57kmath_57kcode_34kscience_deduped--0.8-easy-math-code-final.parquet")
