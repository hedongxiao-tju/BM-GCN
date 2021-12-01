import sys
import torch
import torch.nn.functional as F

from args import parameter_parser
from model import MLP, GCN, BMGCN
from utils import load_data, setup_seed, accuracy

args = parameter_parser()
setup_seed(args.seed, torch.cuda.is_available())
print(args)
print()

acc = []
for repeat in range(10):
    print('-------------------- Repeat {} Start -------------------'.format(repeat))
    # load data
    adj, features, labels, labels_oneHot, train_idx, val_idx, test_idx = load_data(args.dataset, repeat, args.device, args.self_loop)
    print('Data load init finish')
    print('Num nodes: {} | Num features: {} | Num classes: {}'.format(
        adj.shape[0], features.shape[1], labels_oneHot.shape[1] + 1))

    # init model
    model_mlp = MLP(features.shape[1], args.hidden_dim, labels_oneHot.shape[1], args.num_mlp_layers, args.dropout_mlp)
    model_mlp.to(args.device)
    optimizer_mlp = torch.optim.Adam(model_mlp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    print('\nMLP model init finish')

    best_epoch, best_acc = -1, 0.
    # train_time_mlp = time.time()
    for epoch in range(args.epoch_mlp):
        model_mlp.train()
        optimizer_mlp.zero_grad()
        logits = model_mlp(features)
        train_loss = F.nll_loss(torch.log(logits)[train_idx], labels[train_idx])
        train_acc = accuracy(logits[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer_mlp.step()

        model_mlp.eval()
        logits = model_mlp(features)
        val_loss = F.nll_loss(torch.log(logits)[val_idx], labels[val_idx])
        val_acc = accuracy(logits[val_idx], labels[val_idx])
        test_acc = accuracy(logits[test_idx], labels[test_idx])

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model_mlp.state_dict(), 'checkpoint/{}_best_mlp'.format(args.dataset))

        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write("Epoch #{:4d}\tTrain Loss: {:.6f} | Train Acc: {:.4f}".format(epoch, train_loss.item(), train_acc))
        sys.stdout.write(" | Val Loss: {:.6f} | Val Acc: {:.4f} | Test Acc: {:.4f}".format(val_loss.item(), val_acc, test_acc))

    print('\nPre-train MLP best_epoch: {}, best_val_acc: {:.4f}'.format(best_epoch, best_acc))
    model_mlp.load_state_dict(torch.load('checkpoint/{}_best_mlp'.format(args.dataset)))

    # init model
    model_gcn = GCN(in_size=features.shape[1], hidden_size=args.hidden_dim, out_size=labels_oneHot.shape[1],
                    num_layers=args.num_gcn_layers, dropout=args.dropout_gcn)
    model_gcn.to(args.device)
    model_cpgnn = BMGCN(labels_oneHot.shape[1], model_mlp, model_gcn, args.loss_weight, args.enhance, args.device)
    model_cpgnn.to(args.device)

    all_params = model_cpgnn.parameters()
    no_decay = []
    for pname, p in model_cpgnn.named_parameters():
        if pname == 'H' or pname[-4:] == 'bias':
            no_decay += [p]
    params_id = list(map(id, no_decay))
    other_params = list(filter(lambda p: id(p) not in params_id, all_params))
    optimizer_cpgnn = torch.optim.Adam([
        {'params': no_decay},
        {'params': other_params, 'weight_decay': args.weight_decay}],
        lr=args.lr)

    best_epoch, best_acc = -1, 0.
    patience = 0
    running_epoch = args.epoch_gcn
    for epoch in range(args.epoch_gcn):
        model_cpgnn.train()
        optimizer_cpgnn.zero_grad()
        logits, train_loss, _, _, _ = model_cpgnn(features, adj, train_idx, labels[train_idx], labels_oneHot, train_idx)
        train_acc = accuracy(logits[train_idx], labels[train_idx])
        train_loss.backward()
        optimizer_cpgnn.step()

        model_cpgnn.eval()
        logits, val_loss, _, _, _ = model_cpgnn(features, adj, val_idx, labels[val_idx], labels_oneHot, train_idx)
        val_acc = accuracy(logits[val_idx], labels[val_idx])
        test_acc = accuracy(logits[test_idx], labels[test_idx])

        if val_acc >= best_acc:
            best_acc = val_acc
            best_epoch = epoch
            torch.save(model_cpgnn.state_dict(), 'checkpoint/{}_best_BMGCN'.format(args.dataset))
            patience = 0
        else:
            patience += 1
        if patience == args.patience and epoch > 800:
            print('\nEarly stopping at epoch {}'.format(epoch))
            running_epoch = epoch
            break

        sys.stdout.flush()
        sys.stdout.write('\r')
        sys.stdout.write("Epoch #{:4d}\tTrain Loss: {:.6f} | Train Acc: {:.4f}".format(epoch, train_loss.item(), train_acc))
        sys.stdout.write(" | Val Loss: {:.6f} | Val Acc: {:.4f} | Test Acc: {:.4f}".format(val_loss.item(), val_acc, test_acc))

    model_cpgnn.load_state_dict(torch.load('checkpoint/{}_best_BMGCN'.format(args.dataset)))
    model_cpgnn.eval()
    logits, _, H, Q, emb = model_cpgnn(features, adj, test_idx, labels[test_idx], labels_oneHot, train_idx)
    test_acc = accuracy(logits[test_idx], labels[test_idx])
    print('\nBM-GCN best_val_epoch: {}, test_acc: {:.4f}'.format(best_epoch, test_acc))

    print('******************** Repeat {} Done ********************\n'.format(repeat))
    acc.append(round(test_acc.item(), 4))

print('Result: {}'.format(acc))
print('Avg acc: {:.6f}'.format(sum(acc) / 10))
print('\nAll Done!')
