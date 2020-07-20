import torch
import torch.nn as nn
import numpy as np


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=300):
        super().__init__()
        self.items_vectors = nn.Embedding.from_pretrained(torch.from_numpy(np.array([[0.01] * n_factors] * n_items)))
        self.users_vectors = nn.Embedding.from_pretrained(torch.from_numpy(np.array([[0.01] * n_factors] * n_users)))
        self.users_bias = nn.Embedding.from_pretrained(torch.from_numpy(np.array([[0.01] * 1] * n_users)))
        self.items_bias = nn.Embedding.from_pretrained(torch.from_numpy(np.array([[0.01] * 1] * n_items)))
        print(self.items_bias)
        print(nn.Embedding(n_items, 1))
        # self.items_vectors = nn.Embedding(n_items, n_factors)
        # self.users_vectors = nn.Embedding(n_users, n_factors)
        # self.users_bias = nn.Embedding(n_users, 1)
        # self.items_bias = nn.Embedding(n_items, 1)

    def forward(self, user_id, item_id):
        feat_user = self.users_vectors(user_id)
        feat_item = self.items_vectors(item_id)
        bias_user = self.users_bias(user_id)
        bias_item = self.items_bias(item_id)
        result = (feat_user * feat_item).sum(-1) + bias_item.sum(-1) + bias_user.sum(-1)
        return result


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


def write_format(target_list: list):
    return "\t".join([str(i) for i in target_list]) + "\n"


if __name__ == "__main__":
    train_data, test_data, max_min_user, max_min_item = read_original_data(filename="data/u.data")

    # train
    model = MatrixFactorization(n_items=max_min_item[1], n_users=max_min_user[1], n_factors=300)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05, weight_decay=0.01)
    epochs = 50
    for epoch in range(epochs):
        for param_group in optimizer.param_groups:
            if epoch < 5:
                param_group["lr"] = 0.1
            elif epoch < 10:
                param_group["lr"] = 0.05
            elif epoch < 20:
                param_group["lr"] = 0.01
            else:
                param_group["lr"] = 0.002
        loss = 0
        for idx, element in enumerate(train_data):
            if idx > 0 and idx % 5000 == 0:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                print(epoch, idx, loss)
                loss = 0

            user_id, item_id, rating = element
            i = torch.LongTensor([user_id - 1])
            j = torch.LongTensor([item_id - 1])
            rating = torch.FloatTensor([rating])
            # predict
            prediction = model.forward(i, j)
            loss += loss_fn(prediction, rating)
            # for params in model.parameters():
            #     loss_local += 0.01 * torch.norm(params, 1)
            #     loss_local += 0.01 * torch.norm(params, 2)
        # Reset the gradients to 0
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(epoch, idx, loss)

    # test
    mse = 0
    result_file = open("data/mf.test_result", "w", encoding="utf-8")
    for element in test_data:
        user_id, item_id, rating = element
        user_id_function = torch.LongTensor([user_id - 1])
        item_id_function = torch.LongTensor([item_id - 1])
        predict_value = (model.users_vectors(user_id_function) * model.items_vectors(item_id_function)).sum(-1) + model.users_bias(user_id_function).sum(-1) + model.items_bias(item_id_function).sum(-1)
        predict_value = predict_value.detach().numpy().reshape(-1).tolist()[0]
        mse += (rating - predict_value) ** 2
        result_file.write(write_format(target_list=[user_id, item_id, rating, predict_value]))
    result_file.close()
    print(np.sqrt(mse / len(test_data)))
