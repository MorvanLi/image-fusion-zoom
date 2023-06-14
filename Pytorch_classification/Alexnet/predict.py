# -*- coding: utf-8 -*-
"""
@Author  : Morvan Li
@FileName: predict.py
@Software: PyCharm
@Time    : 6/14/23 4:05 PM
"""
import json
import os

import matplotlib.pyplot as plt
import torch
from model import AlexNet
from PIL import Image
from torchvision import transforms, datasets, utils
from tqdm import tqdm
import sys
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize((224, 224)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    data_root = os.path.abspath(os.path.join(os.getcwd(), ".."))  # get data root path
    image_path = os.path.join(data_root, "data_set", "flower_data")  # flower data set path
    assert os.path.exists(image_path), "{} path does not exist.".format(image_path)
    validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                            transform=data_transform)
    val_num = len(validate_dataset)
    batch_size = 4
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8])  # number of workers
    validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                                  batch_size=batch_size,
                                                  shuffle=False,
                                                  num_workers=nw)

    print(f"using {val_num} images for validation.")
    # test_data_iter = iter(validate_loader)
    # test_image, test_label = next(test_data_iter)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = AlexNet(num_classes=5).to(device)

    # load model weights
    weights_path = "./AlexNet.pth"
    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    model.load_state_dict(torch.load(weights_path))

    model.eval()
    y_test = []
    y_pred = []

    with torch.no_grad():
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = model(val_images.to(device))
            pred_y = torch.max(outputs, dim=1)[1].cpu().numpy().tolist()
            y_pred += pred_y
            y_test += val_labels.cpu().numpy().tolist()

    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)

    # 打印混淆矩阵
    print("Confusion Matrix: ")
    print(cm)

    # 画出混淆矩阵
    # ConfusionMatrixDisplay 需要的参数: confusion_matrix(混淆矩阵), display_labels(标签名称列表)
    labels = [val for key, val in class_indict.items()]
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot()
    plt.show()

if __name__ == '__main__':
    main()