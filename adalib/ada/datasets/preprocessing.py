import torchvision.transforms as transforms


def get_transform(kind=None):
    if kind == "mnist32":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    elif kind == "mnist32rgb":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "mnistrgb":
        transform = transforms.Compose(
            [
                transforms.Grayscale(3),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "usps32":
        transform = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
    elif kind == "normalized":
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )
    elif kind == "mnistm":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "svhn":
        transform = transforms.Compose(
            [
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )
    elif kind == "office":
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        transform = transforms.ToTensor()
    return transform
