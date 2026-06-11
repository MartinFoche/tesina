import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.io import read_video
from torchvision.transforms import v2
from torchvision.models.video import r2plus1d_18, R2Plus1D_18_Weights
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

# ==========================================
# 0. REPRODUCIBILIDAD
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 1. DATASET 
# ==========================================
class PesoMuertoDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, train=True):
        self.clip_len = clip_len
        self.train = train
        self.video_files = []
        self.labels = []
        self.class_to_idx = {"Bien": 0, "Mal": 1}

        for label_name, label_idx in self.class_to_idx.items():
            dir_path = os.path.join(root_dir, label_name)
            if not os.path.isdir(dir_path):
                raise FileNotFoundError(f"No se encontró la carpeta {dir_path}")
            for f in sorted(os.listdir(dir_path)):
                if f.lower().endswith(('.mp4', '.avi', '.mov')):
                    self.video_files.append(os.path.join(dir_path, f))
                    self.labels.append(label_idx)

        if len(self.video_files) == 0:
            raise RuntimeError(f"No se encontraron videos en {root_dir}")

        if train:
            self.spatial_transform = v2.Compose([
                v2.Resize((128, 128)),
                v2.RandomResizedCrop(112),
                v2.RandomHorizontalFlip(p=0.5),
                v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ])
        else:
            self.spatial_transform = v2.Compose([
                v2.Resize((128, 128)),
                v2.CenterCrop(112),
            ])

    def __len__(self):
        return len(self.video_files)

    def __getitem__(self, idx):
        video, _, _ = read_video(self.video_files[idx], pts_unit='sec')

        if video.shape[1] > 3:
            video = video[:, :3, :, :]
        elif video.shape[1] == 1:
            video = video.repeat(1, 3, 1, 1)

        total_frames = video.shape[0]
        indices = torch.linspace(0, total_frames - 1, self.clip_len).long()
        video = video[indices, :, :, :]

        video = self.spatial_transform(video)
        video = video.permute(1, 0, 2, 3).float()

        video = video / 255.0
        mean = torch.tensor([0.43216, 0.394666, 0.37645]).view(3, 1, 1, 1)
        std  = torch.tensor([0.22803, 0.22145, 0.216989]).view(3, 1, 1, 1)
        video = (video - mean) / std

        return video, self.labels[idx]

# ==========================================
# PROGRAMA PRINCIPAL
# ==========================================
if __name__ == '__main__':
    from multiprocessing import freeze_support
    freeze_support()
    set_seed(42)

    PATH_DATASET = r"V:\TesinaVideos\Editados"
    DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE   = 16
    NUM_EPOCHS   = 50
    LEARNING_RATE = 0.0003
    NUM_WORKERS  = 4

    # Cargar dataset completo para obtener las etiquetas
    print(f"Cargando dataset desde {PATH_DATASET}...")
    base_dataset = PesoMuertoDataset(PATH_DATASET, train=False)
    labels = base_dataset.labels

    # Split estratificado: garantiza la misma proporción Bien/Mal en train y val
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, val_idx = next(sss.split(np.zeros(len(labels)), labels))

    bien_train = sum(1 for i in train_idx if labels[i] == 0)
    mal_train  = sum(1 for i in train_idx if labels[i] == 1)
    bien_val   = sum(1 for i in val_idx   if labels[i] == 0)
    mal_val    = sum(1 for i in val_idx   if labels[i] == 1)
    print(f"Train: {len(train_idx)} videos (Bien: {bien_train} | Mal: {mal_train})")
    print(f"Val:   {len(val_idx)}  videos (Bien: {bien_val}  | Mal: {mal_val})")

    # Crear subsets con transforms correctos
    train_dataset = Subset(PesoMuertoDataset(PATH_DATASET, train=True),  train_idx)
    val_dataset   = Subset(PesoMuertoDataset(PATH_DATASET, train=False), val_idx)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=True)
    val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=True)

    # Modelo 
    print("Cargando modelo R(2+1)D con pesos de Kinetics-400...")
    weights = R2Plus1D_18_Weights.DEFAULT
    model = r2plus1d_18(weights=weights)
    model.fc = nn.Linear(512, 2)
    model = model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)

    patience = 5
    best_acc = 0.0
    epochs_no_improve = 0

    print(f"Iniciando entrenamiento en {DEVICE}...")
    for epoch in range(NUM_EPOCHS):
        # Entrenamiento
        model.train()
        running_loss = 0.0
        for videos, labels_batch in train_loader:
            videos, labels_batch = videos.to(DEVICE), labels_batch.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(videos)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validación
        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        all_preds, all_labels = [], []
        with torch.no_grad():
            for videos, labels_batch in val_loader:
                videos, labels_batch = videos.to(DEVICE), labels_batch.to(DEVICE)
                outputs = model(videos)
                loss = criterion(outputs, labels_batch)
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels_batch.size(0)
                correct += (predicted == labels_batch).sum().item()
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels_batch.cpu().tolist())

        accuracy     = 100 * correct / total
        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss   = val_loss / len(val_loader)

        # Accuracy por clase
        bien_correct = sum(1 for p, l in zip(all_preds, all_labels) if p == l == 0)
        mal_correct  = sum(1 for p, l in zip(all_preds, all_labels) if p == l == 1)
        bien_total   = all_labels.count(0)
        mal_total    = all_labels.count(1)
        acc_bien = 100 * bien_correct / bien_total if bien_total else 0
        acc_mal  = 100 * mal_correct  / mal_total  if mal_total  else 0

        print(f"Época [{epoch+1:02d}/{NUM_EPOCHS}] "
              f"Loss train/val: {avg_train_loss:.4f}/{avg_val_loss:.4f} | "
              f"Acc: {accuracy:.1f}% (Bien: {acc_bien:.0f}% | Mal: {acc_mal:.0f}%)")

        scheduler.step(accuracy)

        if accuracy > best_acc:
            best_acc = accuracy
            torch.save(model.state_dict(), "mejor_modelo_tesina.pth")
            epochs_no_improve = 0
            print("  >>> Nuevo mejor modelo guardado!")

    print(f"\nEntrenamiento completo! Mejor precisión: {best_acc:.2f}%")
    print("Modelo guardado en: mejor_modelo_tesina.pth")