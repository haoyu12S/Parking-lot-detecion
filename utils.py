from Librarys import *
# Base Configuration class
# Don't use this class directly. Instead, sub-class it and override


class Config:

    name = None

    img_width = None
    img_height = None
    img_channel = None

    batch_size = None

    learning_rate = None
    learning_momentum = 0.9
    weight_decay = None

    shuffle = False

    def __init__(self):
        self.IMAGE_SHAPE = np.array([
            self.img_width, self.img_height, self.img_channel
        ])

    def display(self):
        """Display Configuration values"""
        print("\nConfigurations:")
        for a in dir(self):
            if not a.startswith("__") and not callable(getattr(self, a)):
                print("{:30} {}".format(a, getattr(self, a)))
        print("\n")


class Myconfig(Config):
    """Configuration for training on parking lot detection, overwrite Config"""
    name = 'PLD'

    # Dataset dir
    dataset_dir = './Dataset/ALLdata'

    # Testset dir
    testset_dir = './Dataset/testdata'

    # Log dir
    log_dir = './logs'

    # dir to save checkpoints
    checkpoint_dir = "./checkpoint"

    # Save result to this folder
    num_workers = 1  # number of threads for data loading
    shuffle = True  # shuffle the data set
    batch_size = 2048  # GTX1060 6G Memory
    epochs = 100  # number of epochs to train
    is_train = True  # True for training, False for making prediction
    continue_train = True
    save_model = True  # True for saving the model, False for not saving the model

    n_gpu = 1  # number of GPUs

    learning_rate = 1e-3  # learning rage
    weight_decay = 1e-4  # weight decay

    pin_memory = True  # use pinned (page-locked) memory. when using CUDA, set to True

    is_cuda = torch.cuda.is_available()  # True --> GPU
    num_gpus = torch.cuda.device_count()  # number of GPUs

    dtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type
    imgtype = torch.cuda.FloatTensor if is_cuda else torch.Tensor  # data type
    tartype = torch.cuda.LongTensor if is_cuda else torch.Tensor  # data type


def show_image(sample):
    """Show image with landmarks for a batch of samples."""
    sample = np.array(sample)
    print('There are {} images detect wrong.'.format(sample.shape[0]))
    out = torch.from_numpy(sample)
    out = torchvision.utils.make_grid(out)
    out = out.numpy().transpose((1, 2, 0))
    img = plt.figure()
    plt.imshow(out)
    img.savefig('wrong_image2.png')
    # plt.pause(10)
    # for image in sample:
    #     image = image.transpose((1, 2, 0))
    #     plt.imshow(image)
    #     plt.pause(1)


if __name__ == '__main__':
    conf = Myconfig()
    conf.display()
    print(os.listdir(conf.dataset_dir))
    # var = conf.dtype
    # print(var)
