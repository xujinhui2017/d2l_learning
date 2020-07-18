import torch
import torch.nn as nn


class AutoRec(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, user_id, item_id):
        pass


def read_original_data(filename: str):
    user_list = []
    item_list = []
    matrix_dict = dict()
    with open(filename, "r", encoding="utf-8") as txt_file:
        for idx, line in enumerate(txt_file):
            if idx > 1000:
                break
            user_id, item_id, rating = line.strip().split('\t')[:3]
            user_id = int(user_id)
            item_id = int(item_id)
            rating = int(rating)
            user_list.append(user_id)
            item_list.append(item_id)
            matrix_dict[str(user_id) + "_" + str(item_id)] = rating
    return user_list, item_list, matrix_dict


if __name__ == "__main__":
    pass
