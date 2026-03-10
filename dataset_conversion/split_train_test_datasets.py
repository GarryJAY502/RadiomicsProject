import json
import random
from typing import List


def split_train_test_datasets(all_cases: List,
                              report_file: str,
                              test_ratio: float = 0.2,
                              seed: int = 42,):
    # Handle blacklist filtering and dataset division

    filtered_cases = []

    for case in all_cases:
        filtered_cases.append(case)
        filtered_cases.sort()
    random.seed(seed)
    random.shuffle(filtered_cases)

    num_total = len(filtered_cases)
    num_test = int(num_total * test_ratio)

    train_cases = filtered_cases[num_test:]
    test_cases = filtered_cases[:num_test]

    split_dict = {
        'num_train': len(train_cases),
        'num_test': len(test_cases),
        'num_total': len(all_cases),
        'train': sorted(train_cases),
        'test': sorted(test_cases),
    }

    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(split_dict, f, indent=4)

    return train_cases, test_cases
