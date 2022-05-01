import os
import cv2
import boto3
import logging
import argparse
import tempfile

import torch
import torch.nn as nn
import torch.nn.functional as F


# Custom models must subclass toch.nn.Module and override `forward`
# See: https://pytorch.org/docs/stable/nn.html#torch.nn.Module
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 3 x 100 x 100
        self.conv1 = nn.Conv2d(3, 32, 4, 2, 1)
        # 32 x 49 x 49 
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1)
        # 64 x 24 x 24
        self.conv3 = nn.Conv2d(64, 128, 4, 2, 1)
        # 128 x 12 x 12
        self.conv4 = nn.Conv2d(128, 256, 4, 2, 1)
        # 256 x 6 x 6
        self.conv5 = nn.Conv2d(256, 512, 4, 2, 1)
        # 512 x 3 x 3
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        self.fc1 = nn.Linear(512*3*3, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.dropout1(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def percentage(value):
    return "{: 5.1f}%".format(100.0 * value)


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            logging.info(
                f"Epoch: {epoch} ({percentage(batch_idx / len(train_loader))}) - Loss: {loss.item()}"
            )


def test(args, model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum batch losses
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        f"Test accuracy: {correct}/{len(test_loader.dataset)} ({percentage(correct / len(test_loader.dataset))})"
    )

    # Log metrics for Katib
    logging.info("loss={:.4f}".format(test_loss))
    logging.info("accuracy={:.4f}".format(float(correct) / len(test_loader.dataset)))


def evaluate(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(
                output, target, reduction="sum"
            ).item()  # sum batch losses
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    logging.info(
        f"Test accuracy: {correct}/{len(test_loader.dataset)} ({percentage(correct / len(test_loader.dataset))})"
    )

    # Log metrics for Katib
    logging.info("loss={:.4f}".format(test_loss))
    logging.info("accuracy={:.4f}".format(float(correct) / len(test_loader.dataset)))

    filename = "/home/Output.txt"
    with open(filename, "w") as text_file:
        print(f"accuracy: {float(correct) / len(test_loader.dataset)} | loss: {test_loss} |", file=text_file)

    return filename

def myParser():
    parser = argparse.ArgumentParser(description="PyTorch MNIST Training Job")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="Batch size for training (default: 64)",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1000,
        metavar="N",
        help="Batch size for testing (default: 1000)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="Number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1.0,
        metavar="LR",
        help="Learning rate (default: 1.0)",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.7,
        metavar="M",
        help="Learning rate's decay rate (default: 0.7)",
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        default=False,
        help="Disables CUDA (GPU) training",
    )
    parser.add_argument(
        "--seed", type=int, default=1, metavar="S", help="Random seed (default: 1)"
    )
    parser.add_argument(
        "--log-interval",
        type=int,
        default=10,
        metavar="N",
        help="Number of training batches between status log entries",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Whether to save the trained model",
    )


    args, _ = parser.parse_known_args()
    return args



def read_s3_data(data_path_in_s3, out_path, bucket_name, 
                    AWS_REGION, type='train'):
    #os.makedirs("data", exist_ok=True)
    os.makedirs(os.path.join(out_path, type), exist_ok=True)

    # Access S3
    s3 = boto3.resource('s3', region_name=AWS_REGION) # aws_access_key_id=ACCESS_KEY_ID, aws_secret_access_key=ACCESS_SECRET_KEY)
    my_bucket = s3.Bucket(bucket_name)

    for i, object_summary in enumerate(my_bucket.objects.filter(Prefix=data_path_in_s3)):
        print(i, object_summary.key)

        if i==0:
            continue

        obj = my_bucket.Object(object_summary.key)
        #print(obj.key.split('/')[-1])

        tmp = tempfile.NamedTemporaryFile(prefix='img', delete=False) #, dir='data') #, delete=False)
        #print(tmp.name)
        with open(tmp.name, 'wb') as f:
            obj.download_fileobj(f)
            img = cv2.imread(tmp.name)
            # print(img.shape)
            cv2.imwrite(os.path.join(os.path.join(out_path, type), obj.key.split("/")[-1]), img)
        tmp.close()


def upload_to_s3(bucket_name, AWS_REGION, filename='mnist_model.pth'):
    # Upload weights to s3
    conn_s3 = boto3.client('s3', region_name=AWS_REGION)
    output_path = 'results/' + filename
    conn_s3.upload_file("/home/"+filename, bucket_name, output_path)

def download_weights_from_s3(bucket_name, AWS_REGION, 
                            filename='mnist_model.pth'):
    # Access S3
    s3 = boto3.resource('s3', region_name=AWS_REGION)
    my_bucket = s3.Bucket(bucket_name)

    # Download data on your PC
    obj = my_bucket.Object("results/" + filename)
    obj.download_file(Filename="/home/" + filename)