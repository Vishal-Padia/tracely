import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import numpy as np
import time
from pathlib import Path
from tracely import store

torch.manual_seed(2)
torch.cuda.manual_seed_all(2)

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,10,kernel_size=5,stride=1)
        self.conv2 = nn.Conv2d(10,10,kernel_size=5,stride=1)
        self.pool = nn.MaxPool2d(kernel_size=2,stride=2) #2x2 maxpool
        self.fc1 = nn.Linear(4*4*10,100)
        self.fc2 = nn.Linear(100,10)
    
    def forward(self,x):
        x = F.relu(self.conv1(x)) #24x24x10
        x = self.pool(x) #12x12x10
        x = F.relu(self.conv2(x)) #8x8x10
        x = self.pool(x) #4x4x10    
        x = x.view(-1, 4*4*10) #flattening
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def train_model():
    # Initialize training run with config
    config = {
        "batch_size": 100,
        "validation_split": 0.1,
        "shuffle_dataset": True,
        "random_seed": 2,
        "epochs": 10,
        "learning_rate": 0.01,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

    # Initialize a new run with Tracely
    run_id = store.init_run(project="mnist", run_name="cnn_basic", config=config)

    try:
        # Setup device
        device = torch.device(config["device"])

        # Data loading
        train_ds = datasets.MNIST('../data', train=True, download=True, 
                                transform=transforms.Compose([transforms.ToTensor()]))

        # Creating data indices for training and validation splits
        dataset_size = len(train_ds)
        indices = list(range(dataset_size))
        split = int(np.floor(config["validation_split"] * dataset_size))
        
        if config["shuffle_dataset"]:
            np.random.seed(config["random_seed"])
            np.random.shuffle(indices)
        
        train_indices, val_indices = indices[split:], indices[:split]
        train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
        valid_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

        # Create data loaders
        train_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"], 
                                                sampler=train_sampler)
        validation_loader = torch.utils.data.DataLoader(train_ds, batch_size=config["batch_size"],
                                                    sampler=valid_sampler)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=False, download=True,
                        transform=transforms.Compose([transforms.ToTensor()])),
            batch_size=config["batch_size"], shuffle=True)

        # Model setup
        model = Net().to(device)
        optimizer = optim.SGD(model.parameters(), lr=config["learning_rate"])
        criterion = nn.CrossEntropyLoss()

        # Training tracking variables
        n_train = len(train_loader) * config["batch_size"]
        n_val = len(validation_loader) * config["batch_size"]
        start_time = time.time()

        for epoch in range(config["epochs"]):
            # Training phase
            model.train()
            total_loss = 0
            total_acc = 0
            
            for batch_idx, (images, labels) in enumerate(train_loader):
                images = images.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                output = model(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                total_acc += torch.sum(torch.max(output,dim=1)[1] == labels).item() * 1.0
                
                # Log batch metrics
                if batch_idx % 10 == 0:
                    current_step = epoch * len(train_loader) + batch_idx
                    batch_loss = loss.item()
                    batch_acc = 100.0 * (torch.max(output,dim=1)[1] == labels).sum().item() / len(labels)
                    
                    store.log_metric(
                        run_id=run_id,
                        key="batch_loss",
                        value=batch_loss,
                        step=current_step,
                        timestamp=int(time.time() * 1000)
                    )
                    store.log_metric(
                        run_id=run_id,
                        key="batch_accuracy",
                        value=batch_acc,
                        step=current_step,
                        timestamp=int(time.time() * 1000)
                    )
            
            # Log epoch training metrics
            epoch_train_loss = total_loss / n_train
            epoch_train_acc = 100.0 * total_acc / n_train
            
            store.log_metric(
                run_id=run_id,
                key="train_loss",
                value=epoch_train_loss,
                step=epoch,
                timestamp=int(time.time() * 1000)
            )
            store.log_metric(
                run_id=run_id,
                key="train_accuracy",
                value=epoch_train_acc,
                step=epoch,
                timestamp=int(time.time() * 1000)
            )
            
            # Validation phase
            model.eval()
            total_loss_val = 0
            total_acc_val = 0
            
            with torch.no_grad():
                for images, labels in validation_loader:
                    images = images.to(device)
                    labels = labels.to(device)
                    output = model(images)
                    loss = criterion(output, labels)
                    
                    total_loss_val += loss.item()
                    total_acc_val += torch.sum(torch.max(output,dim=1)[1] == labels).item() * 1.0
            
            # Log validation metrics
            val_loss = total_loss_val / n_val
            val_acc = 100.0 * total_acc_val / n_val
            
            store.log_metric(
                run_id=run_id,
                key="val_loss",
                value=val_loss,
                step=epoch,
                timestamp=int(time.time() * 1000)
            )
            store.log_metric(
                run_id=run_id,
                key="val_accuracy",
                value=val_acc,
                step=epoch,
                timestamp=int(time.time() * 1000)
            )
            
            print(f"Epoch {epoch+1}/{config['epochs']}")
            print(f"Training Loss: {epoch_train_loss:.4f}, Accuracy: {epoch_train_acc:.2f}%")
            print(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
            print("-" * 60)

        # Test phase
        model.eval()
        total_acc = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                labels = labels.to(device)
                output = model(images)
                total_acc += torch.sum(torch.max(output,dim=1)[1] == labels).item() * 1.0

        test_accuracy = 100.0 * total_acc / len(test_loader.dataset)
        print(f"\nTest accuracy: {test_accuracy:.2f}%")

        # Log final test accuracy
        store.log_metric(
            run_id=run_id,
            key="test_accuracy",
            value=test_accuracy,
            step=config["epochs"],
            timestamp=int(time.time() * 1000)
        )

        # Save the model
        checkpoint_path = "checkpoints/best_model.pt"
        Path("checkpoints").mkdir(exist_ok=True)
        torch.save(model.state_dict(), checkpoint_path)
        store.log_artifact(run_id=run_id, name="best_model.pt", path=checkpoint_path)

        # Log total training time
        training_time = time.time() - start_time
        store.log_metric(
            run_id=run_id,
            key="total_training_time",
            value=training_time,
            step=config["epochs"],
            timestamp=int(time.time() * 1000)
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
            traceback_str=traceback.format_exc()
        )
        raise

if __name__ == "__main__":
    train_model()
