import torch
import torch.nn as nn
import numpy as np


class AutoRec(torch.nn.Module):
    def __init__(self, n_users):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_users, 10),
            nn.Sigmoid(),  # 激活函数
            nn.Dropout(0.2),
            nn.Linear(10, n_users),
            nn.Dropout(0.2),
            nn.Sigmoid()
        )
        self.decoder = nn.Linear(n_users, n_users)

    def forward(self, x, is_train=1):
        encoder = self.encoder(x)
        decoder = self.decoder(encoder)
        if is_train == 1:
            decoder = decoder * np.sign(x)
        return decoder


def read_original_data(filename: str):
    train_dict = dict()
    test_dict = dict()
    train_list = []
    test_list = []
    max_user_id, min_user_id, max_item_id, min_item_id = [10] * 4

    with open(filename, "r", encoding="utf-8") as txt_file:
        for idx, line in enumerate(txt_file):
            # if idx > 50:
            #     break
            user_id, item_id, rating, times = line.strip().split('\t')
            user_id = int(user_id)
            item_id = int(item_id)
            rating = int(rating)

            if user_id > max_user_id:
                max_user_id = user_id
            elif user_id < min_user_id:
                min_user_id = user_id

            if item_id > max_item_id:
                max_item_id = item_id
            elif item_id < min_item_id:
                min_item_id = item_id

            train_dict.setdefault(user_id, []).append((item_id, rating))
            if user_id not in test_dict or times > test_dict[user_id][-1]:
                test_dict[user_id] = (item_id, rating, times)

    for user_id in train_dict:
        for info in train_dict[user_id]:
            item_id = info[0]
            if item_id != test_dict[user_id][0]:
                train_list.append((user_id, item_id, info[1]))
            else:
                test_list.append((user_id, item_id, info[1]))
    return train_list, test_list, [min_user_id, max_user_id], [min_item_id, max_item_id]


def generate_matrix(n_items: int, n_users: int, data: list):
    result = [[0] * n_users] * n_items
    for element in data:
        user_id, item_id, rating = element
        result[item_id - 1][user_id - 1] = rating
    return result


def write_format(target_list: list):
    return "\t".join([str(i) for i in target_list]) + "\n"


def evaluate(test_info: list, predict_matrix: list, filename: str):
    mse = 0
    with open(filename, "w", encoding="utf-8") as txt_file:
        for element in test_info:
            user_id, item_id, rating = element
            predict_value = predict_matrix[item_id - 1][user_id - 1]
            mse += (predict_value - rating) ** 2
            txt_file.write(write_format(target_list=[user_id, item_id, rating, predict_value]))

    mse /= len(test_data)
    mse = np.sqrt(mse)
    print(mse)


if __name__ == "__main__":
    train_data, test_data, max_min_user, max_min_item = read_original_data(filename="data/u.data")
    max_user_ids = max_min_user[1]
    max_item_ids = max_min_item[1]
    print(max_user_ids, max_item_ids)
    train_matrix = generate_matrix(n_users=max_user_ids, n_items=max_item_ids, data=train_data)

    # train
    model = AutoRec(n_users=max_user_ids)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)
    epochs = 25
    for i in range(epochs):
        print(i)
        loss = []
        loss_local = 0
        for train_vector in train_matrix:
            prediction = model.forward(torch.FloatTensor([train_vector]), is_train=1)
            loss_local += loss_fn(torch.FloatTensor([train_vector]), prediction)
            for params in model.parameters():
                loss_local += 0.01 * torch.norm(params, 1)
                loss_local += 0.01 * torch.norm(params, 2)
            loss.append(loss_local)
        # Reset the gradients to 0
        optimizer.zero_grad()

        # backpropagate
        [l.backward(retain_graph=True) for l in loss]

        # update weights
        optimizer.step()
        print(loss)

    # predict and evaluate
    predict_info = list()
    for vector in train_matrix:
        prediction = model.forward(torch.FloatTensor([vector]), is_train=0)
        predict_info.append(prediction.detach().numpy().reshape(-1).tolist())
    evaluate(test_info=test_data, predict_matrix=predict_info, filename="data/autorec.test_result")
    pass
