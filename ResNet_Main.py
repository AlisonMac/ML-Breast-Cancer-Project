import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from skimage.io import imread
from sklearn.metrics import roc_auc_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet18
from tqdm import tqdm

# with thanks to Nick Konz for this article that helped me immensely: https://sites.duke.edu/mazurowski/2022/07/13/breast-mri-cancer-detect-tutorial-part1/

print(os.listdir())
dataset_root = Path("d:/")
list(dataset_root.iterdir())
os.chdir(dataset_root)

filename_prefix = "Saved_RNet18_Sample_Images_"


def train_evaluate(learning_rate, weight_decay, train_batchsize, eval_batchsize):
    learning_rate = 10**learning_rate
    weight_decay = 10**weight_decay
    train_batchsize = int(train_batchsize)
    eval_batchsize = int(eval_batchsize)
    # directory where our .png data is (created in the previous post)
    data_dir = ""
    # length in pixels of size of image once resized for the network
    img_size = 224
    # Use GPU
    device = torch.device("cuda")
    print(device)
    # Add data augmentation to improve validation accuracy
    train_transforms = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    eval_transforms = transforms.ToTensor()
    class VdmDataset(Dataset):
        def __init__(self, train=True):
            self.data_dir = data_dir
            self.img_size = img_size
            self.train = train

            # assign labels to data within this Dataset
            self.labels = None
            self.create_labels()
        def create_labels(self):
            # create and store a label (positive/1 or negative/0 for each image)
            # each label is the tuple: (img filename, label number (0 or 1))
            labels = []
            # iterate over each class
            for target, target_label in enumerate(["neg", "pos"]):
                case_dir = os.path.join(self.data_dir, target_label)
                # iterate over all images in the class/case type
                for fname in os.listdir(case_dir):
                    if ".png" in fname:
                        fpath = os.path.join(case_dir, fname)
                        labels.append((fpath, target))
            self.labels = labels

        def normalize(self, img):
            # normalize image pixel values to range [0, 255]
            # img expected to be array

            # convert uint16 -> float
            img = img.astype(float) * 255.0 / img.max()
            # convert float -> unit8
            img = img.astype(np.uint8)
            return img

        def __getitem__(self, idx):
            # required method for accessing data samples
            # returns data with its label
            fpath, target = self.labels[idx]
            # load img from file (png or jpg)
            img_arr = imread(fpath, as_gray=True)
            # normalize image
            img_arr = self.normalize(img_arr)
            # convert to tensor (PyTorch matrix)
            data = torch.from_numpy(img_arr)
            data = data.type(torch.FloatTensor)
            # add image channel dimension (to work with neural network)
            data = Image.fromarray(img_arr)
            # resize image
            transform = train_transforms if self.train else eval_transforms
            data = transform(data)
            return data, target

        def __len__(self):
            # required method for getting size of dataset
            return len(self.labels)
    dataset = VdmDataset()
    print(len(dataset))
    train_fraction = 0.80
    validation_fraction = 0.20
    dataset_size = len(dataset)
    [print(dataset_size)]
    num_train = int(train_fraction * dataset_size)
    num_validation = int(validation_fraction * dataset_size)
    num_test = dataset_size - (num_train + num_validation)
    # this is not int(test_fraction * dataset_size) to account for the uneven split and remaining 2 images unaccounted for
    print(num_train, num_validation, num_test)
    train_dataset, validation_dataset, test_dataset = torch.utils.data.random_split(
        dataset, [num_train, num_validation, num_test])
    train_batchsize = 100
    eval_batchsize = 10  # can be small due to small dataset size
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batchsize,
        shuffle=True
        # images are loaded in random order
    )
    validation_loader = DataLoader(validation_dataset, batch_size=eval_batchsize)
    test_loader = DataLoader(test_dataset, batch_size=eval_batchsize)
    # set random seeds for reproducibility
    seed = 32
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    net = resnet18(pretrained=True)
    num_classes = 2

    # Modify the first convolutional layer to accept 1-channel images
    net.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
    # Modify the last fully connected layer to match the number of output classes
    net.fc = nn.Linear(net.fc.in_features, num_classes)
    net = net.to(device)
    learning_rate = 0.01
    error_minimizer = torch.optim.SGD(
        net.parameters(), lr=learning_rate, weight_decay=1e-4
    )  # using Adam instead of SGD to try and improve accuracy, with weight decay
    # to add regularisation
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        error_minimizer, mode="max", factor=0.1, patience=4, verbose=True
    )
    epochs = 100
    net_final = net

    best_validation_accuracy = 0.0
    # used to pick the best-performing model on the validation set

    # for training visualization later
    train_accs = []
    val_accs = []
    criterion = nn.CrossEntropyLoss()
    for epoch in range(epochs):
        # set network to training mode, so that its parameters can be changed
        net.train()
        # print training info
        print("### Epoch {}:".format(epoch))
        # statistics needed to compute classification accuracy:
        # the total number of image examples trained on
        total_train_examples = 0

        # the number of examples classified correctly
        num_correct_train = 0

        # iterate over the training set once
        for _, (inputs, targets) in tqdm(
            enumerate(train_loader), total=len(train_dataset) // train_batchsize
        ):
            # load the data onto the computation device.
            # inputs are a tensor of shape:
            #   (batch size, number of channels, image height, image width).
            # targets are a tensor of one-hot-encoded class labels for the inputs,
            #   of shape (batch size, number of classes)
            # in other words,
            inputs = inputs.to(device)
            targets = targets.to(device)
            # reset changes (gradients) to parameters
            error_minimizer.zero_grad()
            # get the network's predictions on the training set batch
            predictions = net(inputs)
            # evaluate the error, and estimate
            #   how much to change the network parameters
            loss = criterion(predictions, targets)
            loss.backward()
            # change parameters
            error_minimizer.step()
            # calculate predicted class label
            # the .max() method returns the maximum entries, and their indices;
            # we just need the index with the highest probability,
            #   not the probability itself.
            _, predicted_class = predictions.max(1)
            total_train_examples += predicted_class.size(0)
            num_correct_train += predicted_class.eq(targets).sum().item()
        # get results
        # total prediction accuracy of network on training set
        train_acc = num_correct_train / total_train_examples
        print("Training accuracy: {}".format(train_acc))
        train_accs.append(train_acc)
        # predict on validation set (similar to training set):
        total_val_examples = 0
        num_correct_val = 0

        # switch network from training mode (parameters can be trained)
        #   to evaluation mode (parameters can't be trained)
        net.eval()
        with torch.no_grad():  # don't save parameter changes
            #                      since this is not for training
            for _, (inputs, targets) in tqdm(
                enumerate(validation_loader),
                total=len(validation_dataset) // eval_batchsize,
            ):
                inputs = inputs.to(device)
                targets = targets.to(device)
                predictions = net(inputs)
                _, predicted_class = predictions.max(1)
                total_val_examples += predicted_class.size(0)
                num_correct_val += predicted_class.eq(targets).sum().item()
        # get results
        # total prediction accuracy of network on validation set
        val_acc = num_correct_val / total_val_examples
        print("Validation accuracy: {}".format(val_acc))
        val_accs.append(val_acc)
        scheduler.step(val_acc)
        # Finally, save model if the validation accuracy is the best so far
        if val_acc <= best_validation_accuracy:
            continue

        print("Validation accuracy improved; saving model.")
        net_final = net
        best_validation_accuracy = val_acc
        if os.path.exists("model_best.pth"):
            os.remove("model_best.pth")
        else:
            model_save_path = "model_best.pth"
            torch.save(net_final.state_dict(), model_save_path)
    torch.save(net_final.state_dict(), "net_final.pth")
    total_test_examples = 0
    num_correct_test = 0

    # true and false positive counts
    false_pos_count = 0
    true_pos_count = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    # visualize a random batch of data with examples
    num_viz = 10
    viz_index = random.randint(0, len(test_dataset) // eval_batchsize)
    # see how well the final trained model does on the test set
    with torch.no_grad():  # don't save parameter gradients/changes since this is not for model training
        for batch_index, (inputs, targets) in enumerate(test_loader):
            # make predictions
            inputs = inputs.to(device)
            targets = targets.to(device)
            predictions = net_final(inputs)
            # compute prediction statistics
            _, predicted_class = predictions.max(1)
            total_test_examples += predicted_class.size(0)
            num_correct_test += predicted_class.eq(targets).sum().item()
            # thanks to
            #   https://gist.github.com/the-bass/cae9f3976866776dea17a5049013258d
            confusion_vector = predicted_class / targets
            num_true_pos = torch.sum(confusion_vector == 1).item()
            num_false_pos = torch.sum(confusion_vector == float("inf")).item()
            true_pos_count += num_true_pos
            false_pos_count += num_false_pos
            # Calculate true positives, true negatives, false positives, and false negatives
            true_positives += ((predicted_class == targets) & (predicted_class == 1)).sum().item()
            true_negatives += ((predicted_class == targets) & (predicted_class == 0)).sum().item()
            false_positives += ((predicted_class != targets) & (predicted_class == 1)).sum().item()
            false_negatives += ((predicted_class != targets) & (predicted_class == 0)).sum().item()
            # plot predictions
            if batch_index != viz_index:
                continue

            print("Saving Example Images:")
            num_viz = min(inputs.size(0), num_viz)
            target_labels = targets[:num_viz].tolist()
            classifier_predictions = predicted_class[:num_viz].tolist()
            plot_and_save_imgbatch(
                inputs[:num_viz], filename_prefix, target_labels, classifier_predictions
            )
    # get total results
    # total prediction accuracy of network on test set
    test_acc = num_correct_test / total_test_examples
    print("Test set accuracy: {}".format(test_acc))
    print(
        "{} true positive classifications, {} false positive classifications".format(
            true_pos_count, false_pos_count
        ))
    # Calculate the F1 score and AUC-ROC
    if total_test_examples > 0:
        # Calculate the F1 score
        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)
        f1 = 2 * precision * recall / (precision + recall)
        # Calculate the AUC-ROC
        y_true = np.concatenate(
            (
                np.ones(true_positives),
                np.zeros(true_negatives + false_positives + false_negatives),))
        y_score = np.concatenate(
            (
                np.ones(true_positives),
                np.zeros(true_negatives),
                np.ones(false_positives),
                np.zeros(false_negatives),))
        auc_roc = roc_auc_score(y_true, y_score)
        print("F1 Score: {}".format(f1))
        print("AUC-ROC: {}".format(auc_roc))
        print("Precision: {}".format(precision))
        print("Recall: {}".format(recall))
    else:
        print("Error: No samples found. Please check your data and code.")
    print("F1 Score: {}".format(f1))
    print("AUC-ROC: {}".format(auc_roc))
    return best_validation_accuracy



def generate_unique_filename(basename):
    index = 0
    while True:
        filename = f"{basename}_{index}.png"
        if not Path(filename).is_file():
            return filename
        index += 1


def plot_and_save_imgbatch(
    images, filename_prefix, target_labels, classifier_predictions
):
    num_images = len(images)
    fig, axes = plt.subplots(1, num_images, figsize=(num_images * 3, 3))

    for img, ax, target_label, classifier_prediction in zip(
        images, axes, target_labels, classifier_predictions
    ):
        ax.imshow(img.cpu().permute(1, 2, 0))
        ax.axis("off")

        # Add target labels and classifier predictions to the image
        label_text = f"Target: {target_label}\nPrediction: {classifier_prediction}"
        ax.set_title(label_text, fontsize=8, wrap=True)

    plt.tight_layout()
    unique_filename = generate_unique_filename(filename_prefix)
    plt.savefig(unique_filename)
    plt.close(fig)
    return unique_filename


train_evaluate(-2.11870, -5.09300, 10.02173, 22.0170)