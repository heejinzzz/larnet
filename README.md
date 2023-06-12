# LaRNet

## Pre-requisites
1. Download and enter the project:

```shell
git clone https://github.com/heejinzzz/larnet.git
cd larnet
```

2. Download `Datasets.zip` and `Checkpoints.zip` from [here](https://pan.baidu.com/s/1IwHemRsgUXyTRv09vaRGLQ?pwd=hwbs) and unzip them into the current directory. The structure of the current directory should be:

```
├──larnet
    ├──.git
    ├──Checkpoints
        ├──larnet_for_iemocap.pth
        ├──larnet_for_meld.pth
    ├──Datasets
        ├──IEMOCAP
        ├──MELD
    ├──.gitignore
    ├──combine_loss.py
    ├──config.py
    ├──dataset.py
    ├──larnet.py
    ├──main.py
    ├──README.md
    ├──requirements.txt
    ├──test.py
    ├──train.py
    ├──utils.py
```

3. Install requirements:

```shell
pip install -r requirements.txt
```

Tip: We recommend the use of virtual environment tools such as Anaconda.

## Test
We provide the weights of two models that reach SOTA on the IEMOCAP and MELD datasets respectively. You can directly use them for testing:

```shell
python test.py --dataset {dataset}
```

For example, if you want to test the model which reaches SOTA on the IEMOCAP dataset, then the command will be:

```shell
python test.py --dataset iemocap
```

The full set of optional arguments are:

```shell
--device DEVICE       device that you want to use to run testing, default is 'cuda' if torch.cuda.is_available()
                    else 'cpu'
--dataset_path DATASET_PATH
                    datasets storage path, default is './Datasets'
--num_workers NUM_WORKERS
                    num_workers of Dataloaders, default is 1
--dataset {meld,iemocap}
                    the dataset you want to test on, available options: ['meld', 'iemocap']
--checkpoints_path CHECKPOINTS_PATH
                    model checkpoint files storage path, default is './Checkpoints'
```

## Train
You can also train a new LaRNet model on IEMOCAP or MELD dataset:

```shell
python main.py --dataset {dataset}
```

For example, if you want to train the LaRNet model on the IEMOCAP dataset, then the command will be:

```shell
python main.py --dataset {iemocap}
```

The full set of optional arguments are:

```shell
--device DEVICE       device that you want to use to run training, default is 'cuda' if torch.cuda.is_available()
                    else 'cpu'
--dataset_path DATASET_PATH
                    datasets storage path, default is './Datasets'
--seed SEED           random seed, default is 0
--lr LR               training learning-rate, default is 2e-6
--epoch EPOCH         the number of training epochs, default is 20
--num_workers NUM_WORKERS
                    num_workers of Dataloaders, default is 1
--modal {text,audio,bimodal}
                    the modal you want to use, available options: ['text', 'audio', 'bimodal'], default is
                    'bimodal'
--dataset {meld,iemocap}
                    the dataset you want to train on, available options: ['meld', 'iemocap']
--max_uttrs_num MAX_UTTRS_NUM
                    the max number of utterances per dialog, if the number of utterances in a dialog exceeds
                    max_uttrs_num, then the dialog will be split into multiple segments, ensuring that the
                    utterances in each segment do not exceed max_uttrs_num, default is 60
--uttr_embedding_dim UTTR_EMBEDDING_DIM
                    the embedding dim of one utterance, default is 1280
--uttr_encoder_layers UTTR_ENCODER_LAYERS
                    the number of the uttr_encoder layers, default is 2
--uttrs_encoder_layers UTTRS_ENCODER_LAYERS
                    the number of the uttrs_encoder layers, default is 4
--uttr_encoder_nhead UTTR_ENCODER_NHEAD
                    the number of the heads in multihead-attention of the uttr_encoder, default is 16
--uttrs_encoder_nhead UTTRS_ENCODER_NHEAD
                    the number of the heads in multihead-attention of the uttrs_encoder, default is 16
--projection_expansion_factor PROJECTION_EXPANSION_FACTOR
                    the expansion factor of the projection block, default is 4
--disable_lar_attention
                    don't use LaR-Attention, default is False
--disable_combine_loss
                    don't use Combine Loss , default is False
```

**Tips:**
* As training requires very large GPU memory, we recommend the NVIDIA A100 SXM4 80GB GPU for training.
* Since we consider a whole dialog containing multiple utterances as a sample, the batch size has been fixed to 1, so it is impossible to reduce the memory usage by reducing the batch size. If you are really short of GPU memory, please reduce the memory usage by setting the `--max_uttrs_num` argument to reduce the maximum number of utterances per sample, but please note that this will reduce the accuracy of the model.
