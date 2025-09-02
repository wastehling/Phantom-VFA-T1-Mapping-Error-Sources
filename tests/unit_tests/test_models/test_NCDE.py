import torch
import torchcde
import math

from models.NCDE.NCDE import NeuralCDE


class TestNCDE:

    @staticmethod
    def get_data(num_timepoints = 10):
        ######################
        # Now we need some data.
        # Here we have a simple example which generates some spirals, some going clockwise, some going anticlockwise.
        ######################
        t = torch.linspace(0., 4 * math.pi, num_timepoints)

        start = torch.rand(128) * 2 * math.pi
        x_pos = torch.cos(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
        x_pos[:64] *= -1
        y_pos = torch.sin(start.unsqueeze(1) + t.unsqueeze(0)) / (1 + 0.5 * t)
        x_pos += 0.01 * torch.randn_like(x_pos)
        y_pos += 0.01 * torch.randn_like(y_pos)
        ######################
        # Easy to forget gotcha: time should be included as a channel; Neural CDEs need to be explicitly told the
        # rate at which time passes. Here, we have a regularly sampled dataset, so appending time is pretty simple.
        ######################
        X = torch.stack([t.unsqueeze(0).repeat(128, 1), x_pos, y_pos], dim=2)
        y = torch.zeros(128)
        y[:64] = 1

        perm = torch.randperm(128)
        X = X[perm]
        y = y[perm]

        ######################
        # X is a tensor of observations, of shape (batch=128, sequence=100, channels=3)
        # y is a tensor of labels, of shape (batch=128,), either 0 or 1 corresponding to anticlockwise or clockwise
        # respectively.
        ######################
        return X, y

    def test_NCDE(self, num_epochs=1):
            train_X, train_y = self.get_data()

            ######################
            # input_channels=3 because we have both the horizontal and vertical position of a point in the spiral, and time.
            # hidden_channels=8 is the number of hidden channels for the evolving z_t, which we get to choose.
            # output_channels=1 because we're doing binary classification.
            ######################
            model = NeuralCDE(input_channels=3, parameters={'model': {'CDE': {'hidden_channels': 100},
                                                                      'initial': {'input_channels': 3, 'hidden_channels': 100},
                                                                      'initial_type': 'Initial',
                                                                      'readout_type': 'Readout',
                                                                      'readout': {'input_channels': 100,
                                                                                  'hidden_channels': 100,
                                                                                  'output_channels': 1,
                                                                                  'depth': 3,
                                                                                  'dropout': 0,
                                                                                  'use_batchnorm': True,
                                                                                  'parallel': False}}}, device=None)
            optimizer = torch.optim.Adam(model.parameters())

            ######################
            # Now we turn our dataset into a continuous path. We do this here via Hermite cubic spline interpolation.
            # The resulting `train_coeffs` is a tensor describing the path.
            # For most problems, it's probably easiest to save this tensor and treat it as the dataset.
            ######################
            train_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(train_X)

            train_dataset = torch.utils.data.TensorDataset(train_coeffs, train_y)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=32)
            for epoch in range(num_epochs):
                for batch in train_dataloader:
                    batch_coeffs, batch_y = batch
                    pred_y = model(batch_coeffs).squeeze(-1)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(pred_y, batch_y)
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad()
                print('Epoch: {}   Training loss: {}'.format(epoch, loss.item()))

            test_X, test_y = self.get_data()
            test_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(test_X)
            pred_y = model(test_coeffs).squeeze(-1)
            binary_prediction = (torch.sigmoid(pred_y) > 0.5).to(test_y.dtype)
            prediction_matches = (binary_prediction == test_y).to(test_y.dtype)
            proportion_correct = prediction_matches.sum() / test_y.size(0)
            print('Test Accuracy: {}'.format(proportion_correct))


