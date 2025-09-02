import logging
import torch


def linear_act_norm_dropout(input_channels, output_channels, use_batchnorm, dropout_ratio):
    """
    function to build a linear layer with activation, batchnorm and dropout
    Args:
        input_channels: amount of input channels
        output_channels: amount of output channels
        use_batchnorm: boolean to use batchnorm
        dropout_ratio: dropout ratio

    Returns:
        module_list: list of modules

    """
    module_list = torch.nn.ModuleList()
    module_list.extend([torch.nn.Linear(input_channels, output_channels)])  # fully connected 1
    if use_batchnorm:
        module_list.extend([torch.nn.BatchNorm1d(output_channels)])
    module_list.extend([torch.nn.ReLU()])
    if dropout_ratio > 0:
        module_list.extend([torch.nn.Dropout(dropout_ratio)])
    return module_list


def build_module_list(depth, input_channels, hidden_channels, output_channels, dropout,
                      use_batchnorm):
    """
    function to build a list of modules
    Args:
        depth: amount of layers
        input_channels: amount of input channels
        hidden_channels: amount of hidden channels
        output_channels: amount of output channels
        dropout: dropout ratio
        use_batchnorm: boolean to use batchnorm

    Returns:
        module_list: list of modules

    """
    # build model
    module_list = torch.nn.ModuleList()

    assert depth > 1, 'assuming minimum depth of 2'

    # input to hidden
    if depth == 2:
        hidden_channels = output_channels

    module_list.extend(linear_act_norm_dropout(input_channels, hidden_channels,
                                               use_batchnorm, dropout))

    # hidden state
    if depth > 2:
        for _ in range(depth - 2):
            module_list.extend(linear_act_norm_dropout(hidden_channels, hidden_channels,
                                                       use_batchnorm, dropout))

    module_list.extend([torch.nn.Linear(hidden_channels, int(output_channels))])
    # module_list.apply(init_normal)

    return module_list


class Readout(torch.nn.Module):
    """
    class for reading out hidden state into tensor of predefined shape
    """
    def __init__(self, parameters, device):
        """
        Constructor for Readout
        Args:
            parameters: readout parameters
            device: cuda or cpu
        """
        super(Readout, self).__init__()

        # retrieve parameters
        self.device = device
        self.depth = parameters['model']['readout']['depth']  # number of blocks
        self.input_channels = parameters['model']['readout']['input_channels']  # hidden state
        self.hidden_channels = parameters['model']['readout']['hidden_channels']  # hidden state
        self.output_channels = parameters['model']['readout']['output_channels']
        self.dropout = parameters['model']['readout']['dropout']
        self.use_batchnorm = parameters['model']['readout']['use_batchnorm']
        self.parallel = parameters['model']['readout']['parallel']

        # readout
        if not self.parallel:
            module_list = build_module_list(self.depth, self.input_channels, self.hidden_channels, self.output_channels,
                                            self.dropout, self.use_batchnorm)
            self.encoder = torch.nn.Sequential(*module_list).to(device)
        else:
            if self.output_channels >= 1:
                module_list_1 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm)
                self.encoder_1 = torch.nn.Sequential(*module_list_1).to(device)
            if self.output_channels >= 2:
                module_list_2 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm)
                self.encoder_2 = torch.nn.Sequential(*module_list_2).to(device)
            if self.output_channels >= 3:
                module_list_3 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm)
                self.encoder_3 = torch.nn.Sequential(*module_list_3).to(device)
            if self.output_channels >= 4:
                module_list_4 = build_module_list(self.depth, self.input_channels, self.hidden_channels,
                                                  1,
                                                  self.dropout,
                                                  self.use_batchnorm)
                self.encoder_4 = torch.nn.Sequential(*module_list_4).to(device)
            if self.output_channels >= 5:
                raise NotImplementedError('only 4 parallel outputs implemented')

    def forward(self, x):
        """
        forward pass of the readout
        Args:
            x: input tensor

        Returns:
            encoding: output tensor
        """
        if not self.parallel:
            encoding = self.encoder(x)
        else:
            encoding_list = []
            if self.output_channels >= 1:
                encoding_list.append(self.encoder_1(x))
            if self.output_channels >= 2:
                encoding_list.append(self.encoder_2(x))
            if self.output_channels >= 3:
                encoding_list.append(self.encoder_3(x))
            if self.output_channels >= 4:
                encoding_list.append(self.encoder_4(x))
            if self.output_channels >= 5:
                raise NotImplementedError
            encoding = torch.cat(encoding_list, dim=1)
        return encoding



