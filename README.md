# Personalized Federated Learning on Long-Tailed Data via Adversarial Feature Augmentation

This is the code for paper :  **Personalized Federated Learning on Long-Tailed Data via Adversarial Feature Augmentation**.

**Abstract**: Personalized Federated Learning (PFL) aims to learn personalized models for each client based on the knowledge across all clients in a privacy-preserving manner. Existing PFL methods generally assume that the underlying global data across all clients are uniformly distributed without considering the long-tail distribution. The joint problem of data heterogeneity and long-tail distribution in the FL environment is more challenging and severely affects the performance of personalized models. In this paper, we propose a PFL method called Federated Learning with Adversarial Feature Augmentation (FedAFA) to address this joint problem in PFL. FedAFA optimizes the personalized model for each client by producing a balanced feature set to enhance the local minority classes. The local minority class features are generated by transferring the knowledge from the local majority class features extracted by the global model in an adversarial example learning manner. The experimental results on benchmarks under different settings of data heterogeneity and long-tail distribution demonstrate that FedAFA significantly improves the personalized performance of each client compared with the state-of-the-art PFL algorithm.
## Dependencies

* PyTorch >= 1.0.0

* torchvision >= 0.2.1

  

## Parameters

| Parameter     | Description                                              |
| ------------- | -------------------------------------------------------- |
| `dataset`     | Dataset to use. Options: `cifar10`,`cifar100`, `fmnist`. |
| `lr`          | Learning rate of model.                                  |
| `alpha`       | NonIIDness control. Option:`0.9,0.5,0.2`.                |
| `local_bs`    | Local batch size of training.                            |
| `test_bs`     | Test batch size .                                        |
| `num_users`   | Number of clients.                                       |
| `frac`        | the fraction of clients to be sampled in each round.     |
| `epochs`      | Number of communication rounds.                          |
| `local_ep`    | Number of local epochs.                                  |
| `imb_factor`  | Imbalanced control. Options: `0.01`.                     |
| `num_classes` | Number of classes.                                       |
| `gen_ep`      | Number of generation epoch.                              |
| `device`      | Specify the device to run the program.                   |
| `seed`        | The initial seed.                                        |
| `c`           | Balance factor.                                          |
| `p`           | Drop probability.                                        |


## Usage

Here is an example to run FedARN on CIFAR-10 with imb_fartor=0.01:

```
python main.py --dataset=cifar10 \
    --lr=0.01 \
    --alpha=0.2\
    --epochs=500\
    --local_ep=1 \
    --gen_ep=10\
    --num_users=20 \
    --num_classes=10 \
    --imb_factor=0.01\
    --c=0.6\
    --p=0.5
```

