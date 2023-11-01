import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def deep_learning(X, y, preprocessor):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_preprocessed = preprocessor.fit_transform(X_train)
    X_test_preprocessed = preprocessor.transform(X_test)

    # Convert sparse matrices to dense arrays
    X_train_preprocessed = X_train_preprocessed.toarray()
    X_test_preprocessed = X_test_preprocessed.toarray()

    # Convert data to PyTorch tensors
    X_train_tensor = torch.tensor(X_train_preprocessed, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test_preprocessed, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)

    # Create DataLoader for training data
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    # Define model
    class SimpleNN(nn.Module):
        def __init__(self, input_dim):
            super(SimpleNN, self).__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.fc2 = nn.Linear(64, 64)
            self.fc3 = nn.Linear(64, 1)

        def forward(self, x):
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            x = self.fc3(x)
            return x

    input_dim = X_train_preprocessed.shape[1]
    model = SimpleNN(input_dim)

    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.RMSprop(model.parameters(), lr=1)  # Higher initial learning rate

    # Define learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.95)  # 5% reduction per epoch

    # Train model
    num_epochs = 100
    for epoch in range(num_epochs):
        for batch_x, batch_y in train_loader:
            # Zero the parameter gradients
            optimizer.zero_grad()
            # Forward pass
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
        # Step the learning rate scheduler
        scheduler.step()
        # Print loss and learning rate for every epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}')


    # Evaluate model (optional)
    model.eval()
    with torch.no_grad():
        test_predictions = model(X_test_tensor)
        test_loss = criterion(test_predictions, y_test_tensor)
        print(f'Test Loss: {test_loss.item():.4f}')
    
    y_pred = test_predictions.numpy()
    mae_nn = mean_absolute_error(y_test, y_pred)
    rmse_nn = np.sqrt(mean_squared_error(y_test, y_pred))
    r2_nn = r2_score(y_test, y_pred)

    # Now you can compare
    print(f'MAE: Neural Network={mae_nn}')
    print(f'RMSE: Neural Network={rmse_nn}')
    print(f'R2: Neural Network={r2_nn}')

    return (y_test, y_pred)