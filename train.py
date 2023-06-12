import torch
from sklearn.metrics import precision_recall_fscore_support
from config import Config


def train(config: Config, model, dataloader, loss_func, optimizer, schedule=None, ignore_index=-1):
    device = config.device
    if config.dataset == "meld":
        print_interval = 200
    elif config.dataset == "iemocap":
        print_interval = 20
    else:
        raise RuntimeError(f"Unknown Dataset: {config.dataset}")

    model.train()
    batch_num = len(dataloader)

    for batch, (x_text, x_audio, y, sentiment_ids) in enumerate(dataloader):
        x_text, x_audio, y, sentiment_ids = x_text.to(device), x_audio.to(device), y.to(device), sentiment_ids.to(device)
        if not config.use_combine_loss:
            pred = model(x_text, x_audio)
            pred = torch.squeeze(pred, 0)
            y = torch.squeeze(y, 0)
            loss = loss_func(pred, y)
        else:
            features, pred = model(x_text, x_audio)
            features = torch.squeeze(features, 0)
            pred = torch.squeeze(pred, 0)
            y = torch.squeeze(y, 0)
            sentiment_ids = torch.squeeze(sentiment_ids, 0)
            loss = loss_func(features, pred, y, sentiment_ids)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if schedule is not None:
            schedule.step()

        if batch % print_interval == 0:
            correct_num = (torch.argmax(pred, dim=1) == y).type(torch.float).sum().item()
            total_num = (y != ignore_index).type(torch.float).sum().item()
            acc = 0 if total_num == 0 else correct_num / total_num
            print(f"[Batch {batch:>4d}/{batch_num:>4d}]\tLoss: {loss.item():.3f}, Acc: {acc:.3f}")


def validate(config: Config, model, dataloader, loss_func, ignore_index=-1):
    device = config.device

    model.eval()
    batch_num = len(dataloader)
    size = 0
    val_loss = 0
    predict_labels = []
    true_labels = []

    with torch.no_grad():
        for x_text, x_audio, y, sentiment_ids in dataloader:
            x_text, x_audio, y, sentiment_ids = x_text.to(device), x_audio.to(device), y.to(device), sentiment_ids.to(device)
            if not config.use_combine_loss:
                pred = model(x_text, x_audio)
                pred = torch.squeeze(pred, 0)
                y = torch.squeeze(y, 0)
                loss = loss_func(pred, y)
            else:
                features, pred = model(x_text, x_audio)
                features = torch.squeeze(features, 0)
                pred = torch.squeeze(pred, 0)
                y = torch.squeeze(y, 0)
                sentiment_ids = torch.squeeze(sentiment_ids, 0)
                loss = loss_func(features, pred, y, sentiment_ids)
            size += (y != ignore_index).type(torch.float).sum().item()
            if torch.isnan(loss).item():
                batch_num -= 1
            else:
                val_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            for i in range(len(y)):
                if y[i] != ignore_index:
                    predict_labels.append(pred[i].item())
                    true_labels.append(y[i].item())

    correct_num = sum(a == b for a, b in zip(true_labels, predict_labels))
    result = precision_recall_fscore_support(true_labels, predict_labels, average='weighted')
    print(f"Val Loss: {val_loss/batch_num:.4f}, Accuracy: {correct_num/size:.4f}, Precision: {result[0]:.4f}, Recall: {result[1]:.4f}, F1-Score: {result[2]:.4f}")
    return val_loss / batch_num


def test(config: Config, model, dataloader, loss_func, ignore_index=-1):
    device = config.device

    model.eval()
    batch_num = len(dataloader)
    size = 0
    test_loss = 0
    predict_labels = []
    true_labels = []

    with torch.no_grad():
        for x_text, x_audio, y, sentiment_ids in dataloader:
            x_text, x_audio, y, sentiment_ids = x_text.to(device), x_audio.to(device), y.to(device), sentiment_ids.to(device)
            if not config.use_combine_loss:
                pred = model(x_text, x_audio)
                pred = torch.squeeze(pred, 0)
                y = torch.squeeze(y, 0)
                loss = loss_func(pred, y)
            else:
                features, pred = model(x_text, x_audio)
                features = torch.squeeze(features, 0)
                pred = torch.squeeze(pred, 0)
                y = torch.squeeze(y, 0)
                sentiment_ids = torch.squeeze(sentiment_ids, 0)
                loss = loss_func(features, pred, y, sentiment_ids)
            size += (y != ignore_index).type(torch.float).sum().item()
            if torch.isnan(loss).item():
                batch_num -= 1
            else:
                test_loss += loss.item()
            pred = torch.argmax(pred, dim=1)
            for i in range(len(y)):
                if y[i] != ignore_index:
                    predict_labels.append(pred[i].item())
                    true_labels.append(y[i].item())

    result = precision_recall_fscore_support(true_labels, predict_labels, average='weighted')
    return result[2]
