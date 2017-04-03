# [Advantage async actor-critic Algorithms (A3C)](https://arxiv.org/abs/1602.01783) in PyTorch

```
@inproceedings{mnih2016asynchronous,
  title={Asynchronous methods for deep reinforcement learning},
  author={Mnih, Volodymyr and Badia, Adria Puigdomenech and Mirza, Mehdi and Graves, Alex and Lillicrap, Timothy P and Harley, Tim and Silver, David and Kavukcuoglu, Koray},
  booktitle={International Conference on Machine Learning},
  year={2016}
}

```

This repository contains an implementation of Adavantage async Actor-Critic (A3C) in PyTorch based on the original paper by the authors and the pytorch implementation (https://github.com/ikostrikov/pytorch-a3c) by [Ilya Kostrikov](https://github.com/ikostrikov).

A3C is the state-of-art Deep Reinforcement Learning method.


## Dependencies
* Python 2.7
* PyTorch
* gym (OpenAI)
* universe (OpenAI)
* opencv (for env state processing)
* visdom (for visualization)

## Training

### Normal version

```
./train_lstm.sh
```

### LSTM version

```
./test_lstm.sh
```

### Check the loss curves of all threads in http://localhost:8097
![loss_png](./assets/loss.png)


>Continuous versions are still on going.

## References

* [Asynchronous methods for deep reinforcement learning on arXiv](https://arxiv.org/abs/1602.01783)
* [Ilya Kostrikov's implementation](https://github.com/ikostrikov/pytorch-a3c).
