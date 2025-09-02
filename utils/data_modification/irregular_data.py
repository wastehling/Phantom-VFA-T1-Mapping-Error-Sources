"""
Copied from torchcde (https://github.com/patrick-kidger/torchcde):
 - prepare_multi_channel_observations()
 - fill_forward()
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import torch


def prepare_multi_channel_observations(batch_times_channels: list):
    """
    Prepare multi-channel observations for input to the model. This function takes a list of times and channels
    Args:
        batch_times_channels: input is composed of: [[times], [channels]] ith times[i] and measures[i] lists of the
        respective sampling time and measurement of batch element i. First get all the times that the batch element
        was sampled at, across all channels.

    Returns:
        stacked tensor (time, observation, mask) together into a tensor of shape (length, channels).
    """

    times_per_channel = batch_times_channels[0]
    channels = batch_times_channels[1]
    times_over_channels = torch.unique(torch.cat(times_per_channel))
    times_over_channels, indices = times_over_channels.sort()
    filled_channels = []
    observational_masks = []

    # Now add NaNs to each channel where the other channel was sampled.
    for channel, time_per_channel in zip(channels, times_per_channel):
        channel_filled = torch.full_like(times_over_channels, float('nan'))
        for idx, timepoint in enumerate(times_over_channels):
            if timepoint in time_per_channel:
                channel_filled[idx] = channel[time_per_channel == timepoint]
        filled_channels.append(channel_filled)

    # Add observational masks
    for filled_channel in filled_channels:
        observational_mask = (~torch.isnan(filled_channel)).cumsum(dim=0)
        observational_masks.append(observational_mask)

    # Stack (time, observation, mask) together into a tensor of shape (length, channels).
    list_to_be_stacked = [times_over_channels]
    for channel in filled_channels:
        list_to_be_stacked.append(channel)
    for mask in observational_masks:
        list_to_be_stacked.append(mask)

    return torch.stack(list_to_be_stacked).T


def fill_forward(batch_elements: list):
    """
    Fill forward missing values in a batch of elements. This function takes a list of batch elements and fills forward
    Args:
        batch_elements: list of batch elements

    Returns:
        filled_batch_elements: list of filled batch elements

    """
    filled_batch_elements = []
    max_length = max([el.size(0) for el in batch_elements])
    for batch_element in batch_elements:
        try:
            filled_batch_elements.append(torch.cat([batch_element, batch_element[-1].unsqueeze(0).expand(
                max_length - batch_element.size(0), batch_element.size(1))]))
        except IndexError:
            filled_batch_elements.append(torch.cat([batch_element, batch_element[-1].unsqueeze(0).expand(
                max_length - batch_element.size(0))]))
    return filled_batch_elements


def batch_elements(batch_elements: list):
    """
    Batch elements together. This function takes a list of batch elements and stacks them together
    Args:
        batch_elements: list of batch elements

    Returns:
        batch: stacked batch elements

    """
    batch = torch.stack(batch_elements)
    return batch



