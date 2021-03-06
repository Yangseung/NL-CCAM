# from torchvision import transforms
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
from mydataset import dataset as my_dataset
import torch


def data_loader(args, test_path=False, multi=False):
    mean_vals = [0.485, 0.456, 0.406]
    std_vals = [0.229, 0.224, 0.225]


    input_size = int(args.input_size)
    crop_size = int(args.crop_size)
    if args.phase == 'test' or args.phase == 'mask':
        tsfm_train=transforms.Compose([transforms.Resize((224, 224)),
                                       # transforms.CenterCrop(crop_size),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])
    else:
        tsfm_train = transforms.Compose([transforms.Resize(input_size),  #356
                                     transforms.RandomCrop(crop_size), #321
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize(mean_vals, std_vals)
                                     ])

    if args.tencrop == 'True':
        func_transforms = [transforms.Resize(input_size),
                           transforms.TenCrop(crop_size),
                           transforms.Lambda(
                               lambda crops: torch.stack(
                                   [transforms.Normalize(mean_vals, std_vals)(transforms.ToTensor()(crop)) for crop in crops])),
                           ]
    else:
        func_transforms = []
        # print input_size, crop_size
        if input_size == 0 or crop_size == 0:
            pass
        else:
            func_transforms.append(transforms.Resize(input_size))
            func_transforms.append(transforms.CenterCrop(crop_size))

        func_transforms.append(transforms.ToTensor())
        func_transforms.append(transforms.Normalize(mean_vals, std_vals))

    tsfm_test = transforms.Compose(func_transforms)

    if multi == True:
        img_train = my_dataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True, multi=True)
        img_test = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_test, with_path=test_path)

        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.phase == 'test':
        # classification -> croping : train_loader
        # localization -> not cropping : val_loader

        img_train = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_test, with_path=test_path)
        img_test = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_train, with_path=test_path)

        train_loader = DataLoader(img_train, batch_size=32, shuffle=False, num_workers=args.num_workers)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    elif args.phase == 'mask':
        img_train = my_dataset(args.train_list, root_dir=args.img_dir, transform=tsfm_test, with_path=test_path)
        img_test = my_dataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=test_path)

        train_loader = DataLoader(img_train, batch_size=32, shuffle=False, num_workers=args.num_workers)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    else:
        img_train = my_dataset(args.train_list, root_dir=args.img_dir, transform=tsfm_train, with_path=True)
        img_test = my_dataset(args.test_list, root_dir=args.img_dir, transform=tsfm_test, with_path=test_path)

        train_loader = DataLoader(img_train, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        val_loader = DataLoader(img_test, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    return train_loader, val_loader