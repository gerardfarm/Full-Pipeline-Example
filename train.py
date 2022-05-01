from typing import NamedTuple



# ================================================================
#  Download data from S3, train your model and upload weights to S3
# ================================================================
def Training(   msg: str,
                bucket_name: str = 'ali-bucket-gerard',
                data_path_in_s3: str = 'data/subset_images/',
                out_path: str = '/home/data/',
                AWS_REGION: str = 'us-east-1') -> NamedTuple(
                            'My_Output',[('feedback', str)]
                ):
    """ 
    Download images data from s3 on your machine
    Parameters:
        - bucket_name : str, name of the bucket
        - data_path_in_s3: str, path of data on S3
    """

    # It is mandotory to put necessary libraries here
    import torch
    import torch.utils.data
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from torch.optim.lr_scheduler import StepLR
    from torchvision import transforms, datasets

    import os
    os.chdir("/home")
    import sys
    sys.path.append('/home')
    from functions import ( myParser, Net, train, test, 
                            read_s3_data, upload_to_s3 )

    print(msg)

    # Download dataser from s3 database
    read_s3_data(data_path_in_s3, out_path, bucket_name, 
                    AWS_REGION, type='train')

    print("===============================================")
    print("Dataset has been downloaded.")

    args = myParser()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)

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
    odds = list(range(1, len(dataset), 2))
    train_set = torch.utils.data.Subset(dataset, evens)
    test_set = torch.utils.data.Subset(dataset, odds)

    train_loader = DataLoader(train_set, batch_size=args.batch_size,
                                shuffle=True, num_workers=2)
    test_loader = DataLoader(test_set, batch_size=args.test_batch_size,
                                shuffle=False, num_workers=2)

    print("===============================================")
    print("Dataloader has been prepared.")
    
    device = torch.device("cuda" if use_cuda else "cpu")
    model = Net().to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)

    print("===============================================")
    print("model, optimizer and sheduler have been defined.")

    for epoch in range(1, args.epochs + 1):
        print(epoch)
        train(args, model, device, train_loader, optimizer, epoch)
        test(args, model, device, test_loader)
        scheduler.step()

    if True: #args.save_model:
        torch.save(model.state_dict(), "/home/mnist_model.pth")
    
    print("===============================================")
    print("Training Done.")

    upload_to_s3(bucket_name, AWS_REGION, filename='mnist_model.pth')

    print("===============================================")
    print("weights have been uploaded to s3.")

    from collections import namedtuple
    feedback_msg = 'Done! Model is saved on s3.'
    func_output = namedtuple('MyOutput', ['feedback'])
    return func_output(feedback_msg)