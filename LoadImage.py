from utils import *


def get_data(dataset_dir, train, mc):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(28),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(28),
            transforms.CenterCrop(28),
            transforms.ToTensor()
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    print("Loading data...")
    if train:
        image_datasets = datasets.ImageFolder(dataset_dir, data_transforms['train'])

        train_id, val_id = train_test_split(image_datasets, test_size=0.01)

        train_dataloaders = torch.utils.data.DataLoader(train_id, batch_size=mc.batch_size,
                                                        shuffle=True, num_workers=4)
        val_dataloaders = torch.utils.data.DataLoader(val_id, batch_size=mc.batch_size,
                                                      shuffle=True, num_workers=4)

        class_names = image_datasets.classes

        len_data = [len(train_id), len(val_id)]

        print('Get {} training data and {} validation data done.'.format(len_data[0], len_data[1]))

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        print("Loading data done...")

        return train_dataloaders, val_dataloaders, class_names, device, len_data

    else:
        print('Processing data now.....')
        image_datasets = datasets.ImageFolder(dataset_dir, data_transforms['train'])

        len_test = len(image_datasets)

        dataloaders = torch.utils.data.DataLoader(image_datasets, batch_size=4,
                                                  shuffle=True, num_workers=4)
        class_names = image_datasets.classes

        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('Get {} data done.'.format(len_test))

        print("Loading data done...")

        return dataloaders, class_names, device, len_test


# Helper function to show a batch
def show_batch(sample, class_names):
    """Show image with landmarks for a batch of samples."""
    inputs, classes = sample
    title = [class_names[x] for x in classes]
    out = torchvision.utils.make_grid(inputs)
    out = out.numpy().transpose((1, 2, 0))
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.pause(5)


if __name__ == '__main__':
    mc = Myconfig()
    dataset_dir = mc.dataset_dir
    train_dataloaders, val_dataloaders, class_names, device, len_data = get_data(dataset_dir, train=True, mc=mc)
    sample = next(iter(train_dataloaders))

    show_batch(sample, class_names)
