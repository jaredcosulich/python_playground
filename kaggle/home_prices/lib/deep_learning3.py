import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

def deep_learning(X, y, preprocessor, num_models=5):
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1, random_state=42)

    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_val_preprocessed = preprocessor.transform(X_val)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays
    X_train_preprocessed = X_train_preprocessed.toarray()
    X_val_preprocessed = X_val_preprocessed.toarray()
    X_test_preprocessed = X_test_preprocessed.toarray()

    scaler = MinMaxScaler()

    scaler.fit(X_train_preprocessed)

    X_train_normalized = scaler.transform(X_train_preprocessed)
    X_val_normalized = scaler.transform(X_val_preprocessed)
    X_test_normalized = scaler.transform(X_test_preprocessed)

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_normalized, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_val_tensor = torch.tensor(X_val_normalized, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_normalized, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout = nn.Dropout(0.5)
            self.fc2 = nn.Linear(64, 64)
            self.bn2 = nn.BatchNorm1d(64)
            self.fc3 = nn.Linear(64, 64)
            self.bn3 = nn.BatchNorm1d(64)
            self.fc4 = nn.Linear(64, 64)
            self.bn4 = nn.BatchNorm1d(64)
            self.fc5 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = torch.relu(self.bn2(self.fc2(x)))
            x = torch.relu(self.bn3(self.fc3(x)))
            x = torch.relu(self.bn4(self.fc4(x)))
            x = self.fc5(x)
            return x

    input_dim = X_train_preprocessed.shape[1]

    # Create a list to store the predictions from each model
    predictions = []

    for i in range(num_models):
        model = SimpleNN(input_dim)

        # Define loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)

        # Implement Early Stopping
        early_stop_patience = 20
        no_improve_epochs = 0
        best_val_loss = float('inf')
        best_model_state = model.state_dict()

        # Train model
        num_epochs = 10000
        for epoch in range(num_epochs):
            model.train()
            for batch_x, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()

            # Validate on the validation set
            model.eval()
            with torch.no_grad():
                val_predictions = model(X_val_tensor)
                val_loss = criterion(val_predictions, y_val_tensor)
                # print(f'Epoch {epoch} R2: {r2_score(y_val, val_predictions.numpy())}')

            # Check for early stopping
            if val_loss.item() < best_val_loss:
                best_val_loss = val_loss.item()
                best_model_state = model.state_dict()  # Update the best model state
                no_improve_epochs = 0
            else:
                no_improve_epochs += 1

            if no_improve_epochs >= early_stop_patience:
                print(f"Early stopping triggered after {epoch} epochs!")
                model.load_state_dict(best_model_state)  # Restore the best model state
                break

            scheduler.step()

        # Evaluate model (optional)
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test_tensor)
            predictions.append(test_predictions.numpy())

    # Average the predictions from all the models in the ensemble
    y_pred = np.mean(predictions, axis=0)
    r2_nn = r2_score(y_test, y_pred)

    print(f'R2: Neural Network={r2_nn}')

    return (y_test, y_pred)