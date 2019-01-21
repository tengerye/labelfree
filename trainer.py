import copy
import time

import torch
import numpy as np

import params


def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=params.n_epochs):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        assert set(['train', 'eval']) == set(dataloaders.keys()), 'Keys in dataloaders are incorrect!'

        for phase in ['train', 'eval']:
            if phase == 'train':
                optimizer.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0

            # Iterate over data.
            optimizer.zero_grad() # zero the parameter gradients

            for idx, data in enumerate(dataloaders[phase]):

                inputs, labels = data
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()

                        if (idx+1) % params.batch_size == 0:
                            optimizer.step()
                            optimizer.zero_grad()

                # statistics
                running_loss += loss.item() / inputs.size(0)


            print('{} Loss: {:.4f}'.format(phase, running_loss))

            # deep copy the model
            if phase == 'val' and running_loss < best_loss:
                best_loss = running_loss
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

