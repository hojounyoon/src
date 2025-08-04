import os
from keras.utils import image_dataset_from_directory
from config import train_directory, test_directory, image_size, batch_size, validation_split

def _split_data(train_directory, test_directory, batch_size, validation_split):
    print('train dataset:')
    train_dataset, validation_dataset = image_dataset_from_directory(
        train_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="both",
        seed=47
    )
    print('test dataset:')
    test_dataset = image_dataset_from_directory(
        test_directory,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False
    )

    return train_dataset, validation_dataset, test_dataset

def get_datasets():
    train_dataset, validation_dataset, test_dataset = \
        _split_data(train_directory, test_directory, batch_size, validation_split)
    return train_dataset, validation_dataset, test_dataset

def get_transfer_datasets():
    # Your code replaces this by loading the dataset
    # you can use image_dataset_from_directory, similar to how the _split_data function is using it
    # ...
    transfer_train_dir = "dogcat/train"
    transfer_test_dir = "dogcat/test1"
    
    print('transfer train/validation dataset:')
    train_dataset = image_dataset_from_directory(
        transfer_train_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="training",
        seed=47
    )

    validation_dataset = image_dataset_from_directory(
        transfer_train_dir,
        label_mode='categorical',
        color_mode='rgb',
        batch_size=batch_size,
        image_size=image_size,
        validation_split=validation_split,
        subset="validation",
        seed=47
    )

    print('transfer test dataset:')
    if os.path.exists(transfer_test_dir):
        test_dataset = image_dataset_from_directory(
            transfer_test_dir,
            label_mode='categorical',
            color_mode='rgb',
            batch_size=batch_size,
            image_size=image_size,
            shuffle=False
        )
    else:
        test_dataset = validation_dataset

    
    return train_dataset, validation_dataset, test_dataset

