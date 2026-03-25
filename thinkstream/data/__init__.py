import re

STREAM_COLD_START = {
    "annotation_path": "./datasets/streaming_cot_cold_processed_5_20.jsonl",
    "data_path": "./",
}

STREAM_RLVR = {
    "annotation_path": "./datasets/streaming_rlvr_processed.jsonl",
    "data_path": "./",
}

data_dict = {
    "stream_cold_start": STREAM_COLD_START,
    "stream_rlvr": STREAM_RLVR,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list
