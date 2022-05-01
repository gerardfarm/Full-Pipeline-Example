

# ================================================================
# Download weights from s3, evaluate model and upload results to s3
# ================================================================
def Evaluation( msg: str='no feedback', 
                bucket_name: str = 'ali-bucket-gerard',
                data_path_in_s3: str = 'data/subset_images/',
                out_path: str = '/home/data/',
                AWS_REGION: str = 'us-east-1'):

    print(msg)

    import torch
    import torch.utils.data
    from torch.utils.data import DataLoader
    from torchvision import transforms, datasets
    
    import os
    os.chdir("/home")
    import sys
    sys.path.append('/home')
    from functions import (evaluate, Net, read_s3_data, 
                            download_weights_from_s3, upload_to_s3)


    # Download dataser from s3 database
    read_s3_data(data_path_in_s3, out_path, bucket_name, 
                    AWS_REGION, type='test')

    print("===============================================")
    print("Dataset has been downloaded.")

    dataset = datasets.ImageFolder( root=out_path,
                                    transform=transforms.Compose([
                                                transforms.ToTensor(),
                                                transforms.Normalize((0.485, 0.456, 0.406), 
                                                                    (0.229, 0.224, 0.225)),
                                                transforms.Resize((100, 100))
                                        ])
                                    )
    print("===============================================")
    print("Dataset has been prepared.")

    evens = list(range(0, len(dataset), 2))
    eval_set = torch.utils.data.Subset(dataset, evens)

    eval_loader = DataLoader(eval_set, batch_size=100,
                                shuffle=False, num_workers=2)

    print("===============================================")
    print("Dataloader has been prepared.")

    download_weights_from_s3(bucket_name, AWS_REGION, 
                                filename='mnist_model.pth')

    print("===============================================")
    print("Weights are downloaded.")

    device = torch.device("cuda" if False else "cpu")
    model = Net().to(device)
    model = Net()

    model.load_state_dict( torch.load("/home/mnist_model.pth"))

    # Evaluation
    filename = evaluate(model, device, eval_loader)

    print("===============================================")
    print("Evaluation is done.")

    upload_to_s3(bucket_name, AWS_REGION, filename=filename.split('/')[-1])

    print("===============================================")
