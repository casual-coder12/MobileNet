import argparse

from models.model import MobileNet
from data.dataset import prepare_cifar10_dataset, prepare_imagenette_dataset, prepare_oxford_pet_dataset
from utils.trainer import MobileNetTransferTrainer
from utils.visualize import plot_training_hist_dict


def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNet on various datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenette",
        choices=["cifar10", "imagenette", "oxford_pet"],
        help="Dataset to use for training (default: imagenette).",
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="both",
        choices=["w", "m", "both"],
        help="Type of save operation: 'w' for weights, 'm' for entire model, 'both' for both (default: w).",
    )
    parser.add_argument(
        "--w_epochs",
        type=int,
        default=10,
        help="Number of warmup epochs to train (default: 10).",
    )
    parser.add_argument(
        "--f_epochs",
        type=int,
        default=5,
        help="Number of fine-tuning epochs to train (default: 5).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training (default: 64).",
    )
    parser.add_argument(
        "--head_lr",
        type=float,
        default=0.001,
        help="Learning rate for the warmup phase (default: 0.001).",
    )
    parser.add_argument(
            "--fine_tune_lr",
            type=float,
            default=0.00002,
            help="Learning rate for the fine-tuning phase (default: 0.00002).",
        )
    parser.add_argument(
        "--load_checkpoint", "-lc",
        action="store_true",
        help="Flag to load from a checkpoint if available (default: False).",
    )
    return parser.parse_args()

def main():
    """
    Main function to train the MobileNet model on the specified dataset.
    It prepares the dataset, initializes the model and trainer, and starts the training process.
    """
    args = parse_args()
  
    print(f"=== Starting Training on {args.dataset.upper()} Dataset ===")

    # Get the appropriate dataset
    if args.dataset == "cifar10":
        train_data, val_data, test_data = prepare_cifar10_dataset(batch_size=args.batch_size)
        input_shape = (32, 32, 3)
        num_classes = 10
    elif args.dataset == "imagenette":
        train_data, val_data, test_data = prepare_imagenette_dataset(batch_size=args.batch_size)
        input_shape = (224, 224, 3)
        num_classes = 10
    elif args.dataset == "oxford_pet":
        train_data, val_data, test_data = prepare_oxford_pet_dataset(batch_size=args.batch_size)
        input_shape = (224, 224, 3)
        num_classes = 37
    else:
        raise ValueError("Invalid dataset name. Choose 'cifar10', 'imagenette', or 'oxford_pet'.")

    # Instantiate model and trainer wrapper
    model = MobileNet(
        input_shape=input_shape, 
        num_classes=num_classes,
        augmentation=True
        )
    
    trainer = MobileNetTransferTrainer(
        model=model,
        head_lr=args.head_lr,
        fine_tune_lr=args.fine_tune_lr,
        )

    # Train model
    history = trainer.train_transfer(
        train_data=train_data, 
        val_data=val_data, 
        dataset_name=args.dataset, 
        warmup_epochs=args.w_epochs, 
        fine_tune_epochs=args.f_epochs,
        fine_tune_at=100,
        save_type=args.save_type,
        )

    plot_training_hist_dict(history, dataset_name=args.dataset)

    # Quick evaluation on test data
    results = trainer.evaluate(test_data=test_data, dataset_name=args.dataset)
    print("\n" + "="*50)
    print(f"--- Results on Test Data (for detailed evaluation - run evaluate.py) ---")
    print(f"Test Loss: {results['loss']:.4f}")
    print(f"Test Accuracy: {results['accuracy'] * 100:.2f}%")


if __name__ == "__main__":
    main()