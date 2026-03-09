import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    cohen_kappa_score
)

from tqdm import tqdm

from preprocess import RetinaDataset, train_tf, val_tf, get_sampler
from cnn_model import CNN


def main():

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    BATCH_SIZE = 8
    EPOCHS = 40

    CSV_PATH = "train.csv"
    IMG_DIR = "train_images"

    # ===============================
    # DATA
    # ===============================

    df = pd.read_csv(CSV_PATH)

    train_df, val_df = train_test_split(
        df,
        test_size=0.2,
        stratify=df["diagnosis"],
        random_state=42
    )

    train_dataset = RetinaDataset(train_df, IMG_DIR, train_tf)
    val_dataset = RetinaDataset(val_df, IMG_DIR, val_tf)

    sampler = get_sampler(train_df)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=sampler,
        num_workers=2,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    # ===============================
    # MODEL
    # ===============================

    model = CNN().to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    BEST_QWK = 0

    # ===============================
    # TRAINING
    # ===============================

    for epoch in range(EPOCHS):

        print("\n====================")
        print("Epoch:", epoch + 1)
        print("====================")

        # -----------------
        # TRAIN
        # -----------------

        model.train()

        train_bar = tqdm(train_loader, desc="Training")

        for imgs, labels in train_bar:

            imgs = imgs.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

            train_bar.set_postfix(loss=loss.item())

        # -----------------
        # VALIDATION
        # -----------------

        model.eval()

        preds = []
        targets = []

        val_bar = tqdm(val_loader, desc="Validation")

        with torch.no_grad():

            for imgs, labels in val_bar:

                imgs = imgs.to(DEVICE)

                outputs = model(imgs)

                p = torch.argmax(outputs, 1)

                preds.extend(p.cpu().numpy())
                targets.extend(labels.numpy())

        # ===============================
        # METRICS
        # ===============================

        acc = accuracy_score(targets, preds)

        precision = precision_score(
            targets, preds, average="weighted"
        )

        recall = recall_score(
            targets, preds, average="weighted"
        )

        f1 = f1_score(
            targets, preds, average="weighted"
        )

        qwk = cohen_kappa_score(
            targets, preds, weights="quadratic"
        )

        cm = confusion_matrix(targets, preds)

        print("\nAccuracy:", acc)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1:", f1)
        print("QWK:", qwk)

        print("\nConfusion Matrix")
        print(cm)

        print("\nClassification Report")
        print(classification_report(targets, preds))

        # ===============================
        # SAVE BEST MODEL
        # ===============================

        if qwk > BEST_QWK:

            BEST_QWK = qwk

            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "qwk": qwk
            }, "best_cnn_model.pth")

            print("Best CNN model saved")

    # ===============================
    # SAVE FINAL MODEL
    # ===============================

    torch.save(model.state_dict(), "final_cnn_model.pth")

    print("\nCNN training finished")


if __name__ == "__main__":
    main()