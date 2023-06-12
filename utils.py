import random
import numpy as np
import torch
import argparse


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--device", type=str, help="device that you want to use to run training, default is 'cuda' if torch.cuda.is_available() else 'cpu'")
    parser.add_argument("--dataset_path", type=str, help="datasets storage path, default is './Datasets'")

    parser.add_argument("--seed", type=int, help="random seed, default is 0")
    parser.add_argument("--lr", type=float, help="training learning-rate, default is 2e-6")
    parser.add_argument("--epoch", type=int, help="the number of training epochs, default is 20")
    parser.add_argument("--num_workers", type=int, help="num_workers of Dataloaders, default is 1")

    parser.add_argument("--modal", type=str, choices=["text", "audio", "bimodal"], help="the modal you want to use, available options: ['text', 'audio', 'bimodal'], default is 'bimodal'")
    parser.add_argument("--dataset", type=str, choices=["meld", "iemocap"], help="the dataset you want to train on, available options: ['meld', 'iemocap']", required=True)
    parser.add_argument("--max_uttrs_num", type=int, help="the max number of utterances per dialog, if the number of utterances in a dialog exceeds max_uttrs_num, then the dialog will be split into multiple segments, ensuring that the utterances in each segment do not exceed max_uttrs_num, default is 60")
    parser.add_argument("--uttr_embedding_dim", type=int, help="the embedding dim of one utterance, default is 1280")
    parser.add_argument("--uttr_encoder_layers", type=int, help="the number of the uttr_encoder layers, default is 2")
    parser.add_argument("--uttrs_encoder_layers", type=int, help="the number of the uttrs_encoder layers, default is 4")
    parser.add_argument("--uttr_encoder_nhead", type=int, help="the number of the heads in multihead-attention of the uttr_encoder, default is 16")
    parser.add_argument("--uttrs_encoder_nhead", type=int, help="the number of the heads in multihead-attention of the uttrs_encoder, default is 16")
    parser.add_argument("--projection_expansion_factor", type=int, help="the expansion factor of the projection block, default is 4")
    parser.add_argument("--disable_lar_attention", action="store_true", help="don't use LaR-Attention, default is False")

    parser.add_argument("--disable_combine_loss", action="store_true", help="don't use Combine Loss , default is False")

    args = parser.parse_args()
    return args


def save_model_checkpoint(model_state_dict, path, epoch, val_f1):
    filename = f"{path}/epoch@{epoch}_f1@{val_f1:.4f}"
    torch.save(model_state_dict, filename)
    return filename


def check_stop(val_loss_record, test_f1_record, epochs):
    if len(val_loss_record) <= 3:
        return False
    if val_loss_record[-1] > val_loss_record[-2] and val_loss_record[-2]  > val_loss_record[-3] and val_loss_record[-3] > val_loss_record[-4]:
        test_f1 = test_f1_record[-4]
        print("Early stopping takes effect")
        print(f"The model checkpoints of epoch {len(val_loss_record)-3} is chosen as the best model checkpoint")
        print(f"The F1 Score it obtains on the test dataset is: {test_f1:.4f}")
        return True
    if len(val_loss_record) == epochs:
        test_f1 = test_f1_record[-1]
        print("Training Complete")
        print("The model checkpoints of the last epoch is chosen as the best model checkpoint")
        print(f"The F1 Score it obtains on the test dataset is: {test_f1:.4f}")
        return True
    return False
