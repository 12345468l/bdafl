from synthesizers.pattern_synthesizer import PatternSynthesizer
import torch

if __name__ == '__main__':
    # backdoor_batch.inputs[:batch_size] = (1 - mask) * backdoor_batch.inputs[:batch_size] + mask * pattern
    mask = torch.tensor([[[1., 0.],
                          [0., 0.]],

                         [[1., 0.],
                          [0., 0.]],

                         [[1., 0.],
                          [0., 0.]]])
    # mask = torch.zeros(3,32,32)
    # backdoor_batch = torch.randn(32,3,32,32)
    # pattern = torch.randn(3,32,32)
    backdoor_batch = torch.tensor([[[[0, 1],
                                     [2, 3]],
                                    [[4, 5],
                                     [6, 7]],
                                    [[8, 9],
                                     [10, 11]]],
                                   [[[0, 1],
                                     [2, 3]],
                                    [[4, 5],
                                     [6, 7]],
                                    [[8, 9],
                                     [10, 11]]]])
    pattern = torch.tensor([[[0, 1],
                             [2, 3]],

                            [[4, 5],
                             [6, 7]],

                            [[8, 9],
                             [10, 11]]])*(-100)
    backdoor_batch = (1 - mask) * backdoor_batch + mask * pattern
    print( ((1 - mask) * backdoor_batch), mask * pattern)
    print(backdoor_batch)
