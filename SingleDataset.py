from Librarys import *
from transform import four_point_transform


class SingleDataset(Dataset):
    def __init__(self, location_txt, originalimg_PATH, transform=None):
        self.location_txt = np.loadtxt(location_txt)
        self.original_img = cv2.imread(originalimg_PATH)
        self.transform = transform

    def __len__(self):
        return len(self.location_txt)

    def __getitem__(self, item):
        image = four_point_transform(self.original_img, self.location_txt[item][0:8].reshape((4, 2)))
        # print('image type is: ', type(image))
        label = self.location_txt[item][8]
        # sample = {'image': image, 'label': label, 'showimg': image}
        sample = {'image': image, 'label': label}
        if self.transform:
            sample = self.transform(sample)

        return sample


class RescaleS(object):
    """Rescale the image in a sample to a given size.
    Args:
        output size(tuple ot int): Desired output size. If tuple, output
        is matched to output size. If int, smaller of image edges is
        matched to output_size keeping aspect radio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        # image, label, oimage = sample['image'], sample['label'], sample['showimg']
        image, label = sample['image'], sample['label']
        # h, w = image.shape[:2]
        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
            # if h > w:
            #     new_h, new_w = self.output_size * h / w, self.output_size
            # else:
            #     new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        img = transform.resize(image, (new_h, new_w))
        # img = img_as_ubyte(img)

        label = np.array([label])

        # return {'image': img, 'label': label, 'showimg': oimage}
        return {'image': img, 'label': label}


class RandomCropS(object):
    """Crop randomly the image in a sample.

    Args; output_size(tuple or int): Desired ouput size. If int, square
    square crop is made.

    """

    def __init__(self, output_size):
        return

    def __call__(self, sample):
        return


class ToTensorS(object):
    """convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # image, label, oimage = sample['image'], sample['label'], sample['showimg']
        image, label = sample['image'], sample['label']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        image = image.transpose((2, 0, 1))
        # print('totensors image and label type are: ', type(image), type(label))

        # return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label), 'showimg': oimage}
        return {'image': torch.from_numpy(image), 'label': torch.from_numpy(label).long()}
