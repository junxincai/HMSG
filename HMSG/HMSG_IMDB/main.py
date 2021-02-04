import torch
from sklearn.metrics import f1_score
import numpy as np
from utils import load_imdb_raw, EarlyStopping
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
    g, features, labels, num_classes, train_idx, val_idx, test_idx, train_mask, \
    val_mask, test_mask = load_imdb_raw()

    if hasattr(torch, 'BoolTensor'):
        train_mask = train_mask.bool()
        val_mask = val_mask.bool()
        test_mask = test_mask.bool()

    features_m, features_a, features_d = features

    features_a = torch.zeros(features_a.shape[0], 10)
    features_d = torch.zeros(features_d.shape[0], 10)

    features_m = features_m.to(args['device'])
    features_a = features_a.to(args['device'])
    features_d = features_d.to(args['device'])

    features = {'movie': features_m, 'actor': features_a, 'director':features_d}
    
    in_size = {'actor': features_a.shape[1], 'movie': features_m.shape[1], 'director': features_d.shape[1]}

    labels = labels.to(args['device'])
    train_mask = train_mask.to(args['device'])
    val_mask = val_mask.to(args['device'])
    test_mask = test_mask.to(args['device'])

    model = HMSG(meta_paths = [['ma','am'], ['md', 'dm'], ['am'], ['dm']],
                in_size = in_size,
                hidden_size = args['hidden_units'],
                out_size = num_classes,
                aggre_type = 'attention',
                num_heads = args['num_heads'],
                dropout = args['dropout']).to(args['device'])
    g = g.to(args['device'])

    stopper = EarlyStopping(patience=args['patience'])
    loss_fcn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args['lr'],
                                 weight_decay=args['weight_decay'])

    for epoch in range(args['num_epochs']):
        model.train()
        z, logits = model(g, features)

        loss = loss_fcn(logits[train_mask], labels[train_mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_acc, train_micro_f1, train_macro_f1 = score(logits[train_mask], labels[train_mask])
        val_loss, val_acc, val_micro_f1, val_macro_f1, z = evaluate(model, g, features, labels, val_mask, loss_fcn)
        early_stop = stopper.step(val_loss.data.item(), val_acc, model)

        print('Epoch {:d} | Train Loss {:.4f} | Train Micro f1 {:.4f} | Train Macro f1 {:.4f} | '
             'Val Loss {:.4f} | Val Micro f1 {:.4f} | Val Macro f1 {:.4f}'.format(
           epoch + 1, loss.item(), train_micro_f1, train_macro_f1, val_loss.item(), val_micro_f1, val_macro_f1))

        if early_stop:
            break

    stopper.load_checkpoint(model)
    test_loss, test_acc, test_micro_f1, test_macro_f1, z = evaluate(model, g, features, labels, test_mask, loss_fcn)

    emd_imdb, label_imdb = z[test_mask], labels[test_mask]
    np.savetxt('./out/emd_imdb.txt',emd_imdb.cpu())
    np.savetxt('./out/label_imdb.txt', np.array(label_imdb.cpu(), dtype=np.int32))

    print('Test loss {:.4f} | Test Micro f1 {:.4f} | Test Macro f1 {:.4f}'.format(
        test_loss.item(), test_micro_f1, test_macro_f1))

if __name__ == '__main__':
    import argparse
    from utils import setup
    parser = argparse.ArgumentParser('HMSG')
    parser.add_argument('-s', '--seed', type=int, default=1,
                        help='Random seed')
    parser.add_argument('-ld', '--log-dir', type=str, default='results',
                        help='Dir for saving training results')
    args = parser.parse_args().__dict__
    args = setup(args)
    main(args)