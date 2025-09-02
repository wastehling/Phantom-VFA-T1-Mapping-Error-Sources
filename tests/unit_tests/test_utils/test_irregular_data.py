import torch

from utils.data_modification.irregular_data import prepare_multi_channel_observations, fill_forward, batch_elements


class TestIrregularData:

    @staticmethod
    def get_data():
        # Batch of three elements, each of two channels. Each element and channel are sampled at different times, and at
        # a different number of times.
        torch.manual_seed(7)
        t1a, t1b = torch.rand(7).sort().values, torch.rand(5).sort().values
        t2a, t2b = torch.rand(9).sort().values, torch.rand(7).sort().values
        t3a, t3b = torch.rand(8).sort().values, torch.rand(3).sort().values
        x1a, x1b = torch.rand_like(t1a), torch.rand_like(t1b)
        x2a, x2b = torch.rand_like(t2a), torch.rand_like(t2b)
        x3a, x3b = torch.rand_like(t3a), torch.rand_like(t3b)
        # Overall this has irregular sampling, missing data, and variable lengths.
        return t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b

    @staticmethod
    def process_elements(ta, tb, xa, xb):
        # First get all the times that the batch element was sampled at, across all channels.
        t, sort_indices = torch.cat([ta, tb]).sort()
        # Now add NaNs to each channel where the other channel was sampled.
        xa_ = torch.cat([xa, torch.full_like(xb, float('nan'))])[sort_indices]
        xb_ = torch.cat([torch.full_like(xa, float('nan')), xb])[sort_indices]
        # Add observational masks
        maska = (~torch.isnan(xa_)).cumsum(dim=0)
        maskb = (~torch.isnan(xb_)).cumsum(dim=0)
        # Stack (time, observation, mask) together into a tensor of shape (length, channels).
        return torch.stack([t, xa_, xb_, maska, maskb], dim=1)

    def test_prepare_multi_channel_data(self):
        # first prepare data as in example script by P. Kidger
        t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b = self.get_data()
        ref_x1 = self.process_elements(t1a, t1b, x1a, x1b)
        ref_x2 = self.process_elements(t2a, t2b, x2a, x2b)
        ref_x3 = self.process_elements(t3a, t3b, x3a, x3b)

        # then prepare data as in our own version
        x1 = prepare_multi_channel_observations([[t1a, t1b], [x1a, x1b]])
        x2 = prepare_multi_channel_observations([[t2a, t2b], [x2a, x2b]])
        x3 = prepare_multi_channel_observations([[t3a, t3b], [x3a, x3b]])

        # assert they produce equal result
        assert ref_x1.allclose(x1, equal_nan=True)
        assert ref_x2.allclose(x2, equal_nan=True)
        assert ref_x3.allclose(x3, equal_nan=True)

    def test_prepare_single_channel_data(self):
        # first get data as in example script by P. Kidger
        t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b = self.get_data()

        # then prepare data as in our own version
        ref_x1 = prepare_multi_channel_observations([[t1a, t1b], [x1a, x1b]])
        x1 = prepare_multi_channel_observations([[t1a], [x1a]])

        # assert they single channel data has shape we expect
        assert x1.shape == torch.Size([7, 3])

    def test_prepare_triple_channel_data(self):
        # get data
        t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b = self.get_data()

        # then prepare data with triple channel
        x1 = prepare_multi_channel_observations([[t1a, t1b, t2a], [x1a, x1b, x2a]])

        # assert shape is conform expectation
        assert x1.shape == torch.Size([21, 7])

    def test_prepare_multi_channel_data_equal_time(self):
        # first prepare data as in example script by P. Kidger
        t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b = self.get_data()

        # then prepare data as in our own version
        x1 = prepare_multi_channel_observations([[t1a, t1b, t1b], [x1a, x1b, x1b]])

        # assert they produce shape conform expectation
        assert x1.shape == torch.Size([12, 7])

    def test_prepare_batch(self):
        # generate data
        t1a, t1b, t2a, t2b, t3a, t3b, x1a, x1b, x2a, x2b, x3a, x3b = self.get_data()

        # prepare data as in our own version
        x1 = prepare_multi_channel_observations([[t1a, t1b], [x1a, x1b]])
        x2 = prepare_multi_channel_observations([[t2a, t2b], [x2a, x2b]])
        x3 = prepare_multi_channel_observations([[t3a, t3b], [x3a, x3b]])

        # prepare batch as in example by P. Kidger
        max_length = max(x1.size(0), x2.size(0), x3.size(0))

        def ref_fill_forward(a_x):
            return torch.cat([a_x, a_x[-1].unsqueeze(0).expand(max_length - a_x.size(0), a_x.size(1))])

        ref_x1 = ref_fill_forward(x1)
        ref_x2 = ref_fill_forward(x2)
        ref_x3 = ref_fill_forward(x3)
        ref_x = torch.stack([ref_x1, ref_x2, ref_x3])

        # prepare batch with our implementation
        filled_x = fill_forward([x1, x2, x3])
        x = batch_elements(filled_x)
        assert ref_x.allclose(x, equal_nan=True)
