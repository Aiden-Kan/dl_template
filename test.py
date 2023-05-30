import torch
from torch.utils.data import DataLoader

from dataset.dataset import MyData
from model.xxNet import xxNet


def test_net(net, device, img_dir, label_dir):
    # 1. Data prepare and divide
    dataset = MyData(img_dir, label_dir)
    n_test = len(dataset)
    print("测试数据集的长度为：{}".format(n_test))

    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 加载模型参数
    save_model_path = "./checkpoint/"
    net.load_state_dict(torch.load(save_model_path, map_location=device))

    # 测试模式
    net.eval()

    total_test = 0
    for batch in test_loader:
        images, labels = batch['image'], batch['label']
        images = images.to(device=device, dtype=torch.float32)

        label_pred = net(images)

        save_path = "runs/pred/" + dataset.image_path[total_test].split("\\")[-1]
        label_pred.tofile(save_path)
        print("File saved to", save_path)
        total_test += 1


if __name__ == "__main__":
    # 选择设备，有cuda用cuda，没有就用cpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 数据集地址
    img_dir = "./data/test/image"
    label_dir = "./data/test/label"
    # 加载网络
    net = xxNet(input_channels=1, output_channels=1)
    # 将网络拷贝到deivce中
    net.to(device=device)
    # 配置参数开始测试
    test_net(net, device, img_dir, label_dir)
