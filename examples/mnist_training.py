import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import time
from pathlib import Path
from tracely import store


class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(64 * 12 * 12, 128)
        self.fc2 = nn.Linear(128, 10)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * 12 * 12)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


def train_model():
    # Initialize training run
    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 5,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
    }

    # Initialize a new run with Tracely
    run_id = store.init_run(project="mnist", run_name="conv_net_basic", config=config)

    try:
        # Setup device
        device = torch.device(config["device"])

        # Data loading
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

        train_dataset = datasets.MNIST(
            "data", train=True, download=True, transform=transform
        )
        test_dataset = datasets.MNIST("data", train=False, transform=transform)

        train_loader = DataLoader(
            train_dataset, batch_size=config["batch_size"], shuffle=True
        )
        test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

        # Model setup
        model = ConvNet().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=config["learning_rate"])

        # Training loop
        best_accuracy = 0.0
        start_time = time.time()

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)

                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()

                # Log batch metrics
                if batch_idx % 100 == 0:
                    batch_loss = train_loss / (batch_idx + 1)
                    batch_acc = 100.0 * correct / total
                    current_step = epoch * len(train_loader) + batch_idx

                    store.log_metric(
                        run_id=run_id,
                        key="batch_loss",
                        value=batch_loss,
                        step=current_step,
                        timestamp=int(time.time() * 1000),
                    )
                    store.log_metric(
                        run_id=run_id,
                        key="batch_accuracy",
                        value=batch_acc,
                        step=current_step,
                        timestamp=int(time.time() * 1000),
                    )

            # Validation phase
            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for data, target in test_loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    val_loss += criterion(output, target).item()
                    _, predicted = output.max(1)
                    total += target.size(0)
                    correct += predicted.eq(target).sum().item()

            val_accuracy = 100.0 * correct / total
            val_loss = val_loss / len(test_loader)

            # Log epoch metrics
            store.log_metric(
                run_id=run_id,
                key="val_loss",
                value=val_loss,
                step=epoch,
                timestamp=int(time.time() * 1000),
            )
            store.log_metric(
                run_id=run_id,
                key="val_accuracy",
                value=val_accuracy,
                step=epoch,
                timestamp=int(time.time() * 1000),
            )

            # Save best model
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                checkpoint_path = "checkpoints/best_model.pt"
                Path("checkpoints").mkdir(exist_ok=True)
                torch.save(model.state_dict(), checkpoint_path)

                # Log the model artifact
                store.log_artifact(
                    run_id=run_id, name="best_model.pt", path=checkpoint_path
                )

            print(f"Epoch {epoch + 1}/{config['epochs']}")
            print(f"Validation Loss: {val_loss:.4f}")
            print(f"Validation Accuracy: {val_accuracy:.2f}%")

        # Log final metrics
        training_time = time.time() - start_time
        store.log_metric(
            run_id=run_id,
            key="total_training_time",
            value=training_time,
            step=config["epochs"],
            timestamp=int(time.time() * 1000),
        )

        # Successfully complete the run
        store.finalize_run(run_id=run_id, success=True)

    except Exception as e:
        # If something goes wrong, log the error
        import traceback

        store.finalize_run(
            run_id=run_id,
            success=False,
            error=str(e),
            traceback_str=traceback.format_exc(),
        )
        raise


if __name__ == "__main__":
    train_model()
