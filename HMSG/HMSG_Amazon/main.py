import torch
from sklearn.metrics import f1_score
import numpy as np
from utils import load_amazon_raw, EarlyStopping
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, average_precision_score
from model import HMSG


def score(logits, labels):
    _, indices = torch.max(logits, dim=1)
    prediction = indices.long().cpu().numpy()
    labels = labels.cpu().numpy()

    accuracy = (prediction == labels).sum() / len(prediction)
    micro_f1 = f1_score(labels, prediction, average='micro')
    macro_f1 = f1_score(labels, prediction, average='macro')

    return accuracy, micro_f1, macro_f1

def evaluate(model, g, features, labels, mask, loss_func):
    model.eval()
    with torch.no_grad():
        z, logits = model(g, features)
    loss = loss_func(logits[mask], labels[mask])
    accuracy, micro_f1, macro_f1 = score(logits[mask], labels[mask])

    return loss, accuracy, micro_f1, macro_f1, z

def main(args):
    g, features, train_val_test_pos_user_item, train_val_test_neg_user_item = load_amazon_raw()

    user_feats, item_feats = features
    # user_feats = torch.zeros(user_feats.shape[0], 100)
    # item_feats = torch.zeros(item_feats.shape[0], 100)
    user_feats = user_feats.to(args['device'])
    item_feats = item_feats.to(args['device'])
    features = {'user': user_feats, 'item': item_feats}

    in_size = {'user': user_feats.shape[1], 'item': item_feats.shape[1]}

    train_pos_user_item = train_val_test_pos_user_item['train_pos_user_item']
    val_pos_user_item = train_val_test_pos_user_item['val_pos_user_item']
    test_pos_user_item = train_val_test_pos_user_item['test_pos_user_item']

    train_neg_user_item = train_val_test_neg_user_item['train_neg_user_item']
    val_neg_user_item = train_val_test_neg_user_item['val_neg_user_item']
    test_neg_user_item = train_val_test_neg_user_item['test_neg_user_item']

    np.random.shuffle(train_neg_user_item)
    train_neg_user_item = train_neg_user_item[:train_pos_user_item.shape[0]]
    np.random.shuffle(val_neg_user_item)
    val_neg_user_item = val_neg_user_item[:val_pos_user_item.shape[0]]
    np.random.shuffle(test_neg_user_item)
    test_neg_user_item = test_neg_user_item[:test_pos_user_item.shape[0]]

    y_true_test = np.array([1] * len(test_pos_user_item) + [0] * len(test_neg_user_item))

    auc_list = []
    ap_list = []

    model = HMSG(meta_paths=[['ui', 'iu'], ['iu', 'ui'], ['ui'], ['iu']], #
                in_size=in_size,
                hidden_size=args['hidden_units'],
                aggre_type='mean',
                num_heads=args['num_heads'],
                dropout=args['dropout']).to(args['device'])
    g = g.to(args['device'])


    stopper = EarlyStopping(patience=args['patience'])
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        embeddings= model(g, features)
        user_embed, item_embed = embeddings['user'], embeddings['item']

        pos_embedding_user = user_embed[train_pos_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
        pos_embedding_item = item_embed[train_pos_user_item[:, 1]].view(-1, item_embed.shape[1], 1)
        neg_embedding_user = user_embed[train_neg_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
        neg_embedding_item = item_embed[train_neg_user_item[:, 1]].view(-1, item_embed.shape[1], 1)

        pos_out = torch.bmm(pos_embedding_user, pos_embedding_item)#.view(-1, 5)
        # pos_out = torch.mean(pos_out, dim=1)
        neg_out = -torch.bmm(neg_embedding_user, neg_embedding_item)#.view(-1, 5)
        # neg_out = torch.sum(neg_out, dim=1)

        train_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))
        # print(train_loss.item())
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            embeddings= model(g, features)
            user_embed, item_embed = embeddings['user'], embeddings['item']

            pos_embedding_user = user_embed[val_pos_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
            pos_embedding_item = item_embed[val_pos_user_item[:, 1]].view(-1, item_embed.shape[1], 1)
            neg_embedding_user = user_embed[val_neg_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
            neg_embedding_item = item_embed[val_neg_user_item[:, 1]].view(-1, item_embed.shape[1], 1)
        
            pos_out = torch.bmm(pos_embedding_user, pos_embedding_item)
            neg_out = -torch.bmm(neg_embedding_user, neg_embedding_item)
            val_loss = -torch.mean(F.logsigmoid(pos_out) + F.logsigmoid(neg_out))

        early_stop = stopper.step(val_loss.data.item(), model)
        if early_stop:
            break

    stopper.load_checkpoint(model)
    model.eval()

    with torch.no_grad():
        embeddings= model(g, features)
        user_embed, item_embed = embeddings['user'], embeddings['item']

        pos_embedding_user = user_embed[test_pos_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
        pos_embedding_item = item_embed[test_pos_user_item[:, 1]].view(-1, item_embed.shape[1], 1)
        neg_embedding_user = user_embed[test_neg_user_item[:, 0]].view(-1, 1, user_embed.shape[1])
        neg_embedding_item = item_embed[test_neg_user_item[:, 1]].view(-1, item_embed.shape[1], 1)

        pos_out = torch.bmm(pos_embedding_user, pos_embedding_item).flatten()
        neg_out = torch.bmm(neg_embedding_user, neg_embedding_item).flatten()

        pos_proba = torch.sigmoid(pos_out)
        neg_proba = torch.sigmoid(neg_out)
        y_proba_test = list(pos_proba.cpu().numpy()) + list(neg_proba.cpu().numpy())
    auc = roc_auc_score(y_true_test, y_proba_test)
    ap = average_precision_score(y_true_test, y_proba_test)
    print('--------------Link Prediction Test--------------')
    print('AUC = {:.4f}'.format(auc))
    print('AP = {:.4f}'.format(ap))
    np.save('./out/res.npy', [auc, ap])


if __name__ == '__main__':
    import argparse
    from utils import setup
    parser = argparse.ArgumentParser('HMSG')
    parser.add_argument('-s', '--seed', type=int, default=15555,
                        help='Random seed')
    args = parser.parse_args().__dict__
    args = setup(args)
    main(args)