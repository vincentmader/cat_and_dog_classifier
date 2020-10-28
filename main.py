import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from net import Net
from get_dataset import main as get_dataset


def main():

    # setup net
    net = Net()
    optimizer = optim.Adam(
        net.parameters(), lr=1e-3
    )
    loss_function = nn.MSELoss()

    # prepare data
    X, y = get_dataset()
    # separate dataset into training and testing
    VAL_PCT = 0.1
    val_size = int(len(X) * VAL_PCT)
    train_X = X[:-val_size]
    train_y = y[:-val_size]
    test_X = X[-val_size:]
    test_y = y[-val_size:]

    # train
    print('Starting traing...\n')
    BATCH_SIZE, EPOCHS = 100, 10  # if memory issues -> decrease batch size
    for epoch in range(EPOCHS):
        for i in tqdm(range(0, len(train_X), BATCH_SIZE)):
            batch_X = train_X[i:i+BATCH_SIZE].view(-1, 1, 50, 50)
            batch_y = train_y[i:i+BATCH_SIZE]

            net.zero_grad()
            outputs = net(batch_X)
            loss = loss_function(outputs, batch_y)
            loss.backward()
            optimizer.step()
        print()

    # evaluate
    correct, total = 0, 0
    with torch.no_grad():
        for i in tqdm(range(len(test_X))):
            real_class = torch.argmax(test_y[i])
            net_out = net(test_X[i].view(-1, 1, 50, 50))[0]
            predicted_class = torch.argmax(net_out)
            if predicted_class == real_class:
                correct += 1
            total += 1

    print('Accuracy:', round(correct / total, 3))
    torch.save(net, './cat_dog_classifier')


if __name__ == "__main__":
    main()
