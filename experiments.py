import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from resnet_with_block import cifar_resnet32, cifar_se_resnet32,\
   cifar_srm_resnet32, cifar_srm_with_corr_matrix_resnet32, srm_resnet50, se_resnet50, resnet50, oursrm_resnet50, cifar_srm_with_median_resnet32
import cifar10
import resnet_with_block
from plot import plot_fit
from Trainer import SRMTrainer
import torch
from torch.utils.data import DataLoader
from train_results import FitResult
import imagenet


def print_with_plot(fit_result: FitResult):
    print(fit_result)
    plot_fit(fit_result)


def run_model(data_name, model_name):
    if data_name=="cifar":
        # parameters:
        num_classes = 10
        batch_size = 128
        epochs_count = 100
        num_workers = 8
        # load data cifar:
        data_dir = 'cifar10'
        dl_train = DataLoader(dataset=cifar10.get_datasets(data_dir)['train'], batch_size=batch_size,
                              num_workers=num_workers)
        dl_test = DataLoader(dataset=cifar10.get_datasets(data_dir)['val'], batch_size=batch_size,
                             num_workers=num_workers)
        if model_name == "srm_with_corr":
            model = cifar_srm_with_corr_matrix_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            fit_result = srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count,
                                 checkpoint_filename=model_name)
            print_with_plot(fit_result)
        elif model_name == "srm_with_median_and_corr_matrix":
            model = resnet_with_block.cifar_srm_with_median_and_corr_matrix_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "srm_with_median":
            model = cifar_srm_with_median_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "srm":
            model = cifar_srm_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "se":
            model = cifar_se_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "resnet":
            model = cifar_resnet32(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [70, 80], 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
    else:
        # parametes:
        num_classes = 1000
        batch_size = 64
        epochs_count = 100
        num_workers = 16
        # load data imagenet:
        data_dir = 'imagenet'
        dl_train = DataLoader(dataset=imagenet.get_datasets(data_dir)['train'], batch_size=batch_size,
                                       num_workers=num_workers)
        dl_test = DataLoader(dataset=imagenet.get_datasets(data_dir)['val'], batch_size=batch_size,
                                      num_workers=num_workers)
        if model_name == "oursrm":
            model = oursrm_resnet50(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = StepLR(optimizer, 30, 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "srm":
            model = srm_resnet50(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = StepLR(optimizer, 30, 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "se":
            model = se_resnet50(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = StepLR(optimizer, 30, 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))
        elif model_name == "resnet":
            model = resnet50(num_classes=num_classes)
            optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9,
                                  weight_decay=1e-4)
            scheduler = StepLR(optimizer, 30, 0.1)
            loss_fn = nn.CrossEntropyLoss()
            srm = SRMTrainer(model, loss_fn, optimizer, scheduler)
            print(srm.fit(dl_train=dl_train, dl_test=dl_test, num_epochs=epochs_count))


if __name__ == '__main__':
    run_model("cifar", "srm_with_corr")




