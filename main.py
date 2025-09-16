import argparse
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from sklearn.model_selection import train_test_split

from src.config import (
    MODELS_DIR, DEVICE, VISION_EPOCHS, VISION_LR, VISION_IMG_SIZE,
    SENSOR_EPOCHS, SENSOR_LR, RANDOM_STATE
)
from src.data.load_vision import load_vision_datasets
from src.data.load_sensors import load_sensor_windows
from src.models.vision_cnn import VisionCNN
from src.models.sensor_rf import train_sensor_rf
from src.models.sensor_lstm import SensorLSTM
from src.utils.plotting import plot_confusion_matrix


# -----------------------------
# Vision training
# -----------------------------
def train_vision(args):
    train_loader, val_loader, num_classes = load_vision_datasets()
    model = VisionCNN(num_classes=num_classes).to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=VISION_LR)

    for epoch in range(1, VISION_EPOCHS + 1):
        # Train
        model.train()
        total, correct, loss_sum = 0, 0, 0.0
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() * x.size(0)
            preds = logits.argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)
        train_acc = correct / total
        train_loss = loss_sum / total

        # Validate
        model.eval()
        with torch.no_grad():
            total, correct, loss_sum = 0, 0, 0.0
            for x, y in val_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = model(x)
                loss = criterion(logits, y)
                loss_sum += loss.item() * x.size(0)
                preds = logits.argmax(1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        val_loss = loss_sum / total
        print(f"[Vision][Epoch {epoch}] train_loss={train_loss:.4f} acc={train_acc:.3f} | "
              f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

    out = MODELS_DIR / "vision_cnn.pth"
    torch.save(model.state_dict(), out)
    print(f"[OK] Saved vision model: {out}")


# -----------------------------
# Sensor training
# -----------------------------
def train_sensor(args):
    Xw, yw = load_sensor_windows()

    if args.model == "rf":
        clf, report, cm = train_sensor_rf(Xw, yw)
        from joblib import dump
        out = MODELS_DIR / "sensor_rf.joblib"
        dump(clf, out)
        print(report)
        plot_confusion_matrix(
            cm, classes=["normal", "fault"],
            title="Sensor RF Confusion",
            save_path="docs/confusion_rf.png",
            show=True
        )

    elif args.model == "lstm":
        X_train, X_test, y_train, y_test = train_test_split(
            Xw, yw, test_size=0.2, random_state=RANDOM_STATE, stratify=yw
        )
        X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)
        y_train = torch.tensor(y_train, dtype=torch.long).to(DEVICE)
        X_test = torch.tensor(X_test, dtype=torch.float32).to(DEVICE)
        y_test = torch.tensor(y_test, dtype=torch.long).to(DEVICE)

        n_features = X_train.shape[2]
        model = SensorLSTM(n_features=n_features, n_classes=2).to(DEVICE)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=SENSOR_LR)

        for epoch in range(1, SENSOR_EPOCHS + 1):
            model.train()
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                preds = logits.argmax(1)
                train_acc = (preds == y_train).float().mean().item()

                model.eval()
                val_logits = model(X_test)
                val_loss = criterion(val_logits, y_test).item()
                val_acc = (val_logits.argmax(1) == y_test).float().mean().item()

            print(f"[LSTM][Epoch {epoch}] loss={loss.item():.4f} acc={train_acc:.3f} | "
                  f"val_loss={val_loss:.4f} acc={val_acc:.3f}")

        out = MODELS_DIR / "sensor_lstm.pth"
        torch.save(model.state_dict(), out)
        print(f"[OK] Saved LSTM model: {out}")


# -----------------------------
# Vision prediction
# -----------------------------
def predict_vision(args):
    model_path = MODELS_DIR / "vision_cnn.pth"
    num_classes = 4
    model = VisionCNN(num_classes=num_classes).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()

    if args.image and Path(args.image).exists():
        tfm = transforms.Compose([
            transforms.Resize(VISION_IMG_SIZE),
            transforms.ToTensor(),
        ])
        img = Image.open(args.image).convert("RGB")
        x = tfm(img).unsqueeze(0).to(DEVICE)
    else:
        x = torch.rand(1, 3, *VISION_IMG_SIZE).to(DEVICE)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)
        pred = int(prob.argmax(1)[0].item())
        conf = float(prob.max().item())
    print(f"Predicted class: {pred} | confidence: {conf:.3f}")


# -----------------------------
# Sensor prediction
# -----------------------------
def predict_sensor(args):
    Xw, _ = load_sensor_windows()

    if args.model == "rf":
        from joblib import load
        rf_path = MODELS_DIR / "sensor_rf.joblib"
        if not rf_path.exists():
            print("RF model not found. Train first.")
            return
        clf = load(rf_path)
        X = Xw[:1].reshape(1, -1)
        pred = int(clf.predict(X)[0])
        print(f"RF prediction: {pred}")

    elif args.model == "lstm":
        lstm_path = MODELS_DIR / "sensor_lstm.pth"
        if not lstm_path.exists():
            print("LSTM model not found. Train first.")
            return
        X = torch.tensor(Xw[:1], dtype=torch.float32).to(DEVICE)
        model = SensorLSTM(n_features=X.shape[2], n_classes=2).to(DEVICE)
        model.load_state_dict(torch.load(lstm_path, map_location=DEVICE))
        model.eval()
        with torch.no_grad():
            pred = int(model(X).argmax(1)[0].item())
        print(f"LSTM prediction: {pred}")


# -----------------------------
# CLI entrypoint
# -----------------------------
def main():
    p = argparse.ArgumentParser(description="Welding Maintenance Demo (PyTorch)")
    sub = p.add_subparsers(dest="cmd", required=True)

    tv = sub.add_parser("train-vision")
    tv.set_defaults(func=train_vision)

    ts = sub.add_parser("train-sensor")
    ts.add_argument("--model", choices=["rf", "lstm"], default="rf")
    ts.set_defaults(func=train_sensor)

    pv = sub.add_parser("predict-vision")
    pv.add_argument("--image", type=str, default=None)
    pv.set_defaults(func=predict_vision)

    ps = sub.add_parser("predict-sensor")
    ps.add_argument("--model", choices=["rf", "lstm"], default="rf")
    ps.set_defaults(func=predict_sensor)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
