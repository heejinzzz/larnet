import argparse
import torch
from torch.utils.data import DataLoader
from dataset import MELD_Dataset, IEMOCAP_Dataset
from larnet import LaRNet
from config import Config
from sklearn.metrics import precision_recall_fscore_support


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, help="device that you want to use to run testing, default is 'cuda' if torch.cuda.is_available() else 'cpu'")
    parser.add_argument("--dataset_path", type=str, help="datasets storage path, default is './Datasets'")
    parser.add_argument("--num_workers", type=int, help="num_workers of Dataloaders, default is 1")
    parser.add_argument("--dataset", type=str, choices=["meld", "iemocap"], help="the dataset you want to test on, available options: ['meld', 'iemocap']", required=True)
    parser.add_argument("--checkpoints_path", type=str, help="model checkpoint files storage path, default is './Checkpoints'")
    args = parser.parse_args()

    checkpoint_path = "./Checkpoints"
    if args.checkpoints_path is not None:
        checkpoint_path = args.checkpoints_path

    config = Config()
    config.dataset = args.dataset
    if args.device is not None:
        config.device = args.device
    if args.dataset_path is not None:
        config.dataset_path = args.dataset_path
    if args.num_workers is not None:
        config.num_workers = args.num_workers
    config.check()
    if args.dataset == "meld":
        config.modal = "text"
        config.num_classes = 7
        config.encode_text_len = 256
        config.sup_con_loss_temperature = 0.25
        config.sup_con_loss_base_temperature = 0.25
        config.sup_con_loss_weight = 0.1
        test_dataset = MELD_Dataset(split="test", config=config)
        checkpoint_file = f"{checkpoint_path}/larnet_for_meld.pth"
    elif args.dataset == "iemocap":
        config.modal = "bimodal"
        config.num_classes = 6
        config.encode_text_len = 196
        config.sup_con_loss_temperature = 0.07
        config.sup_con_loss_base_temperature = 0.07
        config.sup_con_loss_weight = 1.0
        test_dataset = IEMOCAP_Dataset(split="test", config=config)
        checkpoint_file = f"{checkpoint_path}/larnet_for_iemocap.pth"
    else:
        raise RuntimeError(f"Unknown Dataset: {args.dataset}")
    test_dataloader = DataLoader(test_dataset, batch_size=1, num_workers=config.num_workers)
    print("Read dataset succeed.")

    device = config.device
    model = LaRNet(config).to(device)
    print("Create model succeed.")
    model.load_state_dict(torch.load(checkpoint_file))
    print("Load model weights succeed.")

    print("Testing...")
    model.eval()
    predict_labels = []
    true_labels = []
    with torch.no_grad():
        for x_text, x_audio, y, _ in test_dataloader:
            x_text, x_audio, y = x_text.to(device), x_audio.to(device), y.to(device)
            _, pred = model(x_text, x_audio)
            pred = torch.squeeze(pred, 0)
            y = torch.squeeze(y, 0)
            pred = torch.argmax(pred, dim=1)
            for i in range(len(y)):
                if y[i] != -1:
                    predict_labels.append(pred[i].item())
                    true_labels.append(y[i].item())

    result = precision_recall_fscore_support(true_labels, predict_labels, average='weighted')
    print("Test Finish.")
    print("[Test Result]")
    print(f"Precision: {result[0]:.4f}, Recall: {result[1]:.4f}, Weighted F1-Score: {result[2]:.4f}")
