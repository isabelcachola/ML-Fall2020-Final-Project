import torch
import torch.nn.functional as F
import numpy as np
import csv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def approx_train_acc_and_loss(model, train_data, train_labels):
    idxs = np.random.choice(len(train_data), len(train_data), replace=False)
    x = torch.from_numpy(train_data[idxs].astype(np.float32))#.to(device)
    y = torch.from_numpy(train_labels[idxs].astype(np.int))#.to(device)
    logits = model(x).cpu()
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(train_labels[idxs], y_pred.numpy()), loss.item()


def dev_acc_and_loss(model, dev_data, dev_labels):
    x = torch.from_numpy(dev_data.astype(np.float32))
    y = torch.from_numpy(dev_labels.astype(np.int))
    logits = model(x).cpu()
    loss = F.cross_entropy(logits, y)
    y_pred = torch.max(logits, 1)[1]
    return accuracy(dev_labels, y_pred.numpy()), loss.item()


def accuracy(y, y_hat):
    return (y == y_hat).astype(np.float).mean()

def train_pytorch(args, model, train_data, train_labels, dev_data, dev_labels):
    # setup metric logging. It's important to log your loss!!
    log_f = open(args.log_file, 'w')
    fieldnames = ['step', 'train_loss', 'train_acc', 'dev_loss', 'dev_acc']
    logger = csv.DictWriter(log_f, fieldnames)
    logger.writeheader()

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)


    for step in range(args.train_steps):
        i = np.random.choice(train_data.shape[0], size=args.batch_size, replace=False)
        x = torch.from_numpy(train_data[i].astype(np.float32)).to(device)
        y = torch.from_numpy(train_labels[i].astype(np.int)).to(device)

        # Forward pass: Get logits for x
        logits = model(x)
        # Compute loss
        loss = F.cross_entropy(logits, y)
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # every 100 steps, log metrics
        hist_dev_acc = []
        if step % 100 == 0:
            train_acc, train_loss = approx_train_acc_and_loss(model,
                                                              train_data,
                                                              train_labels)
            dev_acc, dev_loss = dev_acc_and_loss(model, dev_data, dev_labels)
            hist_dev_acc.append(dev_acc)

            step_metrics = {
                'step': step, 
                'train_loss': loss.item(), 
                'train_acc': train_acc,
                'dev_loss': dev_loss,
                'dev_acc': dev_acc
            }

            print(f'On step {step}: Train loss {train_loss} | Dev acc is {dev_acc}')
            logger.writerow(step_metrics)
            

    # close the log file
    log_f.close()
    # save model
    print(f'Done training. Saving model at models/simple-ff.torch')
    torch.save(model, 'models/simple-ff.torch')

def test_pytorch(test_data, test_labels):
        model = torch.load('models/simple-ff.torch', map_location=device)
        model.device = device
        predictions_file = "preds/simple-ff-preds.txt"
        preds = []
        for test_ex in test_data:
            x = torch.from_numpy(test_ex.astype(np.float32))
            # Make the x look like it's in a batch of size 1
            x = x.view(1, -1)
            logits = model(x)
            pred = torch.max(logits, 1)[1]
            preds.append(pred.item())
        print(f'Done making predictions! Storing them in {predictions_file}')
        preds = np.array(preds)
        np.savetxt(predictions_file, preds, fmt='%d')
        return preds
