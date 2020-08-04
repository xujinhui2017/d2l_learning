import torch
import torch.nn as nn
import numpy as np


class NeuMF(torch.nn.Module):
    def __init__(self, n_users, n_items, nums_hiddens, n_factors=10):
        super().__init__()
        self.items_mf = nn.Embedding(n_items, n_factors)
        self.users_mf = nn.Embedding(n_users, n_factors)
        self.users_neu = nn.Embedding(n_users, n_factors)
        self.items_neu = nn.Embedding(n_items, n_factors)
        torch.nn.init.normal_(self.users_mf.weight, 0.1)
        torch.nn.init.normal_(self.items_mf.weight, 0.1)
        torch.nn.init.normal_(self.users_neu.weight, 0.1)
        torch.nn.init.normal_(self.items_neu.weight, 0.1)

        self.mlp = nn.Sequential()
        for idx, nums_hidden in enumerate(nums_hiddens):
            if idx == 0:
                self.mlp.add_module("linear_{}".format(idx), nn.Linear(2 * n_factors, nums_hidden))
                self.mlp.add_module("activation_{}".format(idx), nn.Sigmoid())
                self.mlp.add_module("dropout_{}".format(idx), nn.Dropout(0.2))
            else:
                self.mlp.add_module("linear_{}".format(idx), nn.Linear(nums_hiddens[idx - 1], nums_hidden))
                self.mlp.add_module("activation_{}".format(idx), nn.Sigmoid())
                self.mlp.add_module("dropout_{}".format(idx), nn.Dropout(0.2))

    def forward(self, user_id, item_id):
        user_mf = self.users_mf(user_id)
        item_mf = self.items_mf(item_id)
        user_neu = self.users_neu(user_id)
        item_neu = self.items_neu(item_id)
        gmf = user_mf * item_mf
        input_vector = torch.cat((user_neu, item_neu), 1)
        mlp = self.mlp(input_vector)
        combine_result = torch.cat((gmf, mlp), 1)
        return combine_result.sum(-1)


def read_original_data(filename: str):
    train_dict = dict()
    test_dict = dict()
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

    pos_neg_train = dict()
    all_items = set([i for i in range(1, max_item_id + 1)])
    for user_id in train_dict:
        pos_neg_train[user_id] = {
            "pos": [],
            "neg": []
        }
        test_item = 0
        for info in train_dict[user_id]:
            item_id = info[0]
            if item_id != test_dict[user_id][0]:
                pos_neg_train[user_id]["pos"].append(item_id)
            else:
                test_item = item_id
                
        candidates = list(all_items - set(pos_neg_train[user_id]["pos"]) - {test_item})
        test_set = set([test_item] + np.random.choice(candidates, int(len(candidates) / 3), replace=False))
        test_dict[user_id] = {
            "pos": [test_item],
            "neg": list(test_set - set(test_item))
        }
        pos_neg_train[user_id]["neg"] = list(set(candidates) - test_set)

    return pos_neg_train, test_dict, [min_user_id, max_user_id], [min_item_id, max_item_id]


def bpr_loss(positive, negative):
    distances = positive - negative
    loss_local = -torch.log(torch.sigmoid(distances))
    return loss_local


def write_format(target_list: list):
    return "\t".join([str(i) for i in target_list]) + "\n"


def evaluate_auc():
    auc = 0
    for user_id_local in test_data:
        test_pos_item = test_data[user_id]["pos"]
        test_neg_item = test_data[user_id]["neg"]
        neg_score_local = []
        pos_score_local = []
        for item_id_local in test_pos_item:
            pos_score_local = model.forward(torch.LongTensor([user_id_local - 1]), torch.LongTensor([item_id_local - 1]))
        for item_id_local in test_neg_item:
            neg_score_local += [model.forward(torch.LongTensor([user_id_local - 1]), torch.LongTensor([item_id_local - 1]))]
        auc += np.average([pos_score_local > i for i in neg_score_local])
    auc = auc / len(train_data)
    print("test_AUC:", auc)


if __name__ == "__main__":
    train_data, test_data, max_min_user, max_min_item = read_original_data(filename="data/u.data")
    max_user_ids = max_min_user[1]
    max_item_ids = max_min_item[1]
    print(max_min_user, max_min_item)
    model = NeuMF(n_users=max_user_ids, n_items=max_item_ids, n_factors=10, nums_hiddens=[10, 10])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002, weight_decay=0.01)
    epochs = 50
    
    for epoch in range(epochs):
        loss = 0
        loss_count = 0
        for idx, user_id in enumerate(train_data):
            pos_info = train_data[user_id]["pos"]
            neg_info = train_data[user_id]["neg"]
            pos_score = []
            neg_score = []
            for item_id in pos_info:
                pos_score_single = model.forward(torch.LongTensor([user_id - 1]), torch.LongTensor([item_id - 1]))
                neg_idx = np.random.randint(0, len(neg_info) - 1)
                neg_item_id = torch.LongTensor([neg_info[neg_idx] - 1])
                neg_score_single = model.forward(torch.LongTensor([user_id - 1]), neg_item_id)
                
                loss += bpr_loss(positive=pos_score_single, negative=neg_score_single)
                loss_count += 1
                
                if loss_count > 0 and loss_count % 5000 == 0:
                    print(epoch, loss_count, loss)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    loss = 0

        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
        print(epoch, loss_count, loss)
        evaluate_auc()
        
    # evaluate
    
    pass
