import torch
from utils import get_args, set_seed, check_stop
from config import Config
from train import train, validate, test
from torch.utils.data import DataLoader
from larnet import LaRNet
from dataset import MELD_Dataset, IEMOCAP_Dataset
from transformers import get_cosine_schedule_with_warmup
from combine_loss import CombineLoss


def main(config: Config):
    set_seed(config.seed)

    device = config.device

    if config.dataset == "meld":
        train_dataset = MELD_Dataset(split="train", config=config)
        val_dataset = MELD_Dataset(split="dev", config=config)
        test_dataset = MELD_Dataset(split="test", config=config)
    elif config.dataset == "iemocap":
        train_dataset = IEMOCAP_Dataset(split="train", config=config)
        val_dataset = IEMOCAP_Dataset(split="dev", config=config)
        test_dataset = IEMOCAP_Dataset(split="test", config=config)
    else:
        raise RuntimeError(f"Unknown Dataset: {config.dataset}")

    batch_size = 1
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=True, num_workers=config.num_workers)
    val_dataloader = DataLoader(val_dataset, batch_size, shuffle=True, num_workers=config.num_workers)
    test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True, num_workers=config.num_workers)

    model = LaRNet(config).to(device)

    if not config.use_combine_loss:
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=-1)
    else:
        loss_func = CombineLoss(config)

    epoch = config.epoch
    lr = config.lr
    batch_num = len(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    schedule = get_cosine_schedule_with_warmup(optimizer, epoch * batch_num // 6, epoch * batch_num)

    val_loss_record, test_f1_record = [], []
    for i in range(epoch):
        print(f"Epoch {i + 1}\n-----------------------------------")
        train(config, model, train_dataloader, loss_func, optimizer, schedule)
        val_loss = validate(config, model, val_dataloader, loss_func)
        val_loss_record.append(val_loss)
        test_f1 = test(config, model, test_dataloader, loss_func)
        test_f1_record.append(test_f1)
        print("")
        if check_stop(val_loss_record, test_f1_record, epoch):
            break


if __name__ == "__main__":
    args = get_args()
    config = Config()
    config.apply_args(args)
    config.check()
    print(config)
    main(config)
