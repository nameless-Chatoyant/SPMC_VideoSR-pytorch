from torch import nn
import torch.optim as optim
from torch.autograd import Variable

def train(net, train_iter, eval_iter, criterion):
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train_iter, 0):
            # get the inputs
            inputs, labels = data

            # wrap them in Variable
            inputs, labels = Variable(inputs), Variable(labels)

            optimizer = optim.Adam(net.parameters(), lr=0.0001)

            # initialize network parameters
            nn.init.xavier_uniform(net.weight, gain = 1)
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()

            nn.utils.clip_grad_norm(net.detail_fusion_net.convlstm.parameters(), 2)

            optimizer.step()

            # print statistics
            running_loss += loss.data[0]
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0