import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.mlp import MLPRecommender


def build_dataloader(
    user_ids: torch.tensor,
    item_ids: torch.tensor,
    ratings: torch.tensor,
    batch_size: int = 256,
) -> DataLoader:
    """
    Criar dataloader a partir dos tensores
    """
    dataset = TensorDataset(user_ids, item_ids, ratings)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def train_epoch(
    model: MLPRecommender,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
) -> float:
    """
    Treinamento de uma epoca e retorno da loss media
    """
    model.train()
    total_loss = 0.0

    for user_ids, item_ids, ratings in loader:
        optimizer.zero_grad()
        predictions = model(user_ids, item_ids)
        loss = criterion(predictions, ratings)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate_epoch(
    model: MLPRecommender,
    loader: DataLoader,
    criterion: nn.Module,
) -> float:
    """
    Avaliar o modelo e retornar a loss media
    """
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for user_ids, item_ids, ratings in loader:
            predictions = model(user_ids, item_ids)
            loss = criterion(predictions, ratings)
            total_loss += loss.item()

    return total_loss / len(loader)


def train_with_early_stopping(
    model: MLPRecommender,
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_epochs: int = 50,
    patience: int = 5,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
) -> list[dict]:
    """
    Treinamento com early stopping

    Args:
        patience: para o treino se a val_loss não melhorar apos n epocas
        weight decay: regularizacao L2 para o otimizador

    return: historico de metricas por epoca
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = evaluate_epoch(model, val_loader, criterion)
        current_lr = optimizer.param_groups[0]["lr"]

        scheduler.step(val_loss)

        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "lr": current_lr,
            }
        )

        print(
            f"Época {epoch+1:02d} - "
            f"train_loss: {train_loss:.4f} | "
            f"val_loss: {val_loss:.4f} | "
            f"lr: {current_lr:.6f}",
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping na época {epoch+1}")
            break

    return history
