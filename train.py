import logging

import torch
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset.dataset import MyData
from model.xxNet import xxNet

dir_checkpoint = "./checkpoint/epochs/"  # every epoch
final_checkpoint = '../checkpoints/'  # the last epoch


def eval_net(net, loader, device):
    net.eval()
    n_val = len(loader)  # the number of batch
    tot = 0
    criterion = torch.nn.MSELoss()

    with tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        for batch in loader:
            images, labels = batch['image'], batch['label']

            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            with torch.no_grad():
                label_pred = net(images)

            tot += criterion(label_pred, labels).item()

            pbar.update(images.shape[0])

    # net.train()
    return labels, label_pred, tot / n_val


def train_net(net, device, img_dir, label_dir, epochs=50, batch_size=16, lr=1e-5, val_percent=0.2, save_cp=True):
    # 1. Data prepare and divide
    dataset = MyData(img_dir, label_dir)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(train, batch_size=batch_size, shuffle=False)

    # train or val length
    train_data_size = len(train)
    val_data_size = len(val)
    print("训练数据集的长度为：{}".format(train_data_size))
    print("验证数据集的长度为：{}".format(val_data_size))

    # - prepare tensorboard
    writer = SummaryWriter(comment=f'_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}')
    global_step = 0

    logging.info(f'''Starting training:
            Epochs:          {epochs}
            Batch size:      {batch_size}
            Learning rate:   {lr}
            Training size:   {n_train}
            Validation size: {n_val}
            Checkpoints:     {save_cp}
            Device:          {device.type}
        ''')

    # 2. select optimizer and criterion
    # optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=0.0005)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-4, betas=(0.9, 0.999), eps=1e-8)

    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.6)  # after every step_size, lr = lr * gamma
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 50, 80], gamma=0.6)  # at every milestones, lr = lr * gamma
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=0.)  # after T_max epoch, reset lr
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=10, verbose=True)

    criterion = torch.nn.MSELoss()

    # 3. start train
    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='proj') as pbar:
            total_batch = 0
            for batch in train_loader:
                total_batch = total_batch + 1

                images, labels = batch['image'], batch['label']
                images = images.to(device=device, dtype=torch.float32)
                labels = labels.to(device=device, dtype=torch.float32)

                label_pred = net(images)
                loss = criterion(label_pred, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                if epoch > 0:
                    writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                pbar.update(images.shape[0])  # update batch size forward
                global_step += 1

        # Save checkpoint for every epoch
        torch.save(net.state_dict(), dir_checkpoint + 'epoch_{}.pth'.format(epoch))

        # epoch loss average
        writer.add_scalar('Epoch Loss/train', epoch_loss / total_batch, epoch)

        # 4. Validation
        val_labels_true, val_labels_pred, val_score = eval_net(net, val_loader, device)

        # scheduler.step(val_score)
        scheduler.step()
        logging.info('Validation MSE: {}'.format(val_score))

        # - tensorboard record
        for tag, value in net.named_parameters():
            tag = tag.replace('.', '/')
            writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), epoch)
            writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], epoch)
        writer.add_scalar('Loss/test', val_score, epoch)
        writer.add_images('val label/true', val_labels_true, epoch)
        writer.add_images('val label/pred', val_labels_pred, epoch)

    # 5. Save checkpoints
    if save_cp:
        torch.save(net.state_dict(), final_checkpoint + f'CP_BS_{batch_size}_LR_{lr}_EPOCH_{epochs}.pth')
        logging.info(f'Checkpoint saved !')


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集地址
    img_dir = "data/train/image"
    label_dir = "./data/train/label"
    # 加载网络
    net = xxNet(input_channels=1, output_channels=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 配置参数开始训练
    train_net(net, device, img_dir, label_dir, epochs=4, batch_size=1, lr=1e-5, val_percent=0.25, save_cp=True)
