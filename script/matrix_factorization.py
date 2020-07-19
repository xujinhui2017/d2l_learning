import torch
import torch.nn as nn


class MatrixFactorization(torch.nn.Module):
    def __init__(self, n_users, n_items, n_factors=300):
        super().__init__()

        self.items_vectors = nn.Embedding(n_items, n_factors, sparse=True)
        self.users_vectors = nn.Embedding(n_users, n_factors, sparse=True)
        self.users_bias = nn.Embedding(n_users, 1, sparse=True)
        self.items_bias = nn.Embedding(n_items, 1, sparse=True)

    def forward(self, user_id, item_id):
        feat_user = self.users_vectors(user_id)
        feat_item = self.items_vectors(item_id)
        bias_user = self.users_bias(user_id)
        bias_item = self.items_bias(item_id)
        result = (feat_user * feat_item).sum(-1) + bias_item.sum(-1) + bias_user.sum(-1)
        return result


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
    user_ids, item_ids, matrix_info = read_original_data(filename="data/u.data")
    model = MatrixFactorization(n_items=max(item_ids) + 1, n_users=max(user_ids) + 1, n_factors=300)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    epochs = 5
    for epoch in range(epochs):
        print(epoch)
        loss = 0
        loss0 = 0
        for r, c in zip(user_ids, item_ids):
            i = torch.LongTensor([r])
            j = torch.LongTensor([c])
            if str(r) + "_" + str(c) not in matrix_info:
                continue
            rating = torch.FloatTensor([matrix_info[str(r) + "_" + str(c)]])
            # predict
            prediction = model.forward(i, j)
            loss += loss_fn(prediction, rating)
            loss0 += loss_fn(prediction, rating)
            for params in model.parameters():
                loss += 0.01 * torch.norm(params, 1)
                loss += 0.01 * torch.norm(params, 2)

        # Reset the gradients to 0
        optimizer.zero_grad()

        # backpropagate
        loss.backward()

        # update weights
        optimizer.step()
        print(loss0)
