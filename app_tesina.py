import sys
import torch
from torchvision.models.video import r2plus1d_18
import torch.nn as nn
import cv2
import os
from PyQt5.QtGui import QIcon
from torchvision.io import read_video
from torchvision.transforms import v2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QProgressBar, QWidget, QVBoxLayout,
                             QPushButton, QLabel, QFileDialog)
from PyQt5.QtCore import Qt, QThread, QObject, pyqtSignal, QTimer
from PyQt5.QtGui import QPixmap, QImage

# ------------------ FUNCIÓN PARA CARGAR EL MODELO ------------------
def cargar_modelo(ruta_weights, device):
    model = r2plus1d_18(weights=None)
    model.fc = nn.Linear(512, 2)
    model.load_state_dict(torch.load(ruta_weights, map_location=device))
    model.to(device)
    model.eval()
    return model
# ------------------ WORKER QUE ANALIZA EL VIDEO ------------------
class AnalizadorWorker(QObject):
    terminado = pyqtSignal(str)
    progreso = pyqtSignal(int)

    def __init__(self, ruta_video, model, device):
        super().__init__()
        self.ruta_video = ruta_video
        self.model = model
        self.device = device

    def ejecutar(self):
        try:
            # 1. Cargar video con torchvision
            video, _, _ = read_video(self.ruta_video, pts_unit='sec')
            
            # 2. Convertir a (C, T, H, W)
            video = video.permute(1, 0, 2, 3)
            
            # 3. Asegurar 3 canales RGB
            if video.shape[0] > 3:
                video = video[:3, :, :, :]
            elif video.shape[0] == 1:
                video = video.repeat(3, 1, 1, 1)
            
            # 4. Seleccionar 16 frames uniformemente
            total_frames = video.shape[1]
            indices = torch.linspace(0, total_frames - 1, 16).long()
            video = video[:, indices, :, :]  # (C, T, H, W)
            
            # 5. Transformaciones 
            video = v2.Resize((128, 128), antialias=True)(video)
            video = v2.CenterCrop(112)(video)
            video = video.float() / 255.0
            
            #Mover video a GPU 
            video = video.to(self.device)
            
            # Normalizar con tensores
            mean = torch.tensor([0.43216, 0.394666, 0.37645], device=self.device).view(3, 1, 1, 1)
            std = torch.tensor([0.22803, 0.22145, 0.216989], device=self.device).view(3, 1, 1, 1)
            video = (video - mean) / std
            
            # 6. Agregar dimensión de batch 
            input_tensor = video.unsqueeze(0) 
            
            self.progreso.emit(50)
            
            # 7. Predicción
            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.softmax(output, dim=1)
                prob_bien = probs[0][0].item() * 100
                prob_mal = probs[0][1].item() * 100
                clase = torch.argmax(probs).item()
            
            # 8. Resultado
            if clase == 0:
                resultado_texto = f"Técnica Correcta ✅ ({prob_bien:.1f}%)"
            else:
                resultado_texto = f"Posible riesgo de Lesión ⚠️ ({prob_mal:.1f}%)"
            
            self.progreso.emit(100)
            import time
            time.sleep(0.8)   # pausa artificial para ver la barra completa
            self.terminado.emit(resultado_texto)
            
        except Exception as e:
            print(f"ERROR en el análisis: {e}")
            import traceback
            traceback.print_exc()
            self.terminado.emit(f"Error: {str(e)}")

# ------------------ VENTANA PRINCIPAL (con OpenCV para mostrar video) ------------------
class VentanaPrincipal(QMainWindow):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.hilo = None
        self.cap = None
        self.timer = None

        self.setWindowTitle("Analizador de Peso Muerto - Tesina")
        self.setMinimumSize(800, 600)

        widget_central = QWidget()
        self.setCentralWidget(widget_central)
        layout = QVBoxLayout()
        widget_central.setLayout(layout)

        # Label para mostrar el video
        self.video_label = QLabel()
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumHeight(400)

        self.boton = QPushButton("Seleccionar video y analizar")
        self.boton.clicked.connect(self.iniciar_analisis)

        self.resultado_label = QLabel("Esperando video...")
        self.resultado_label.setAlignment(Qt.AlignCenter)
        self.resultado_label.setStyleSheet("font-size: 18px; padding: 10px;")

        # Barra de progreso (inicialmente oculta)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Analizando... %p%")
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border-radius: 8px;
                text-align: center;
                background-color: #3a3a4e;
                height: 25px;
                color: white;
                font-weight: bold;
            }
            QProgressBar::chunk {
                background-color: #3a86ff;
                border-radius: 8px;
            }
        """)
        self.progress_bar.hide()

        layout.addWidget(self.video_label)
        layout.addWidget(self.boton)
        layout.addWidget(self.progress_bar)
        layout.addWidget(self.resultado_label)

        # Estilo visual
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e2f;
            }
            QWidget {
                background-color: #1e1e2f;
                font-family: "Segoe UI", "Roboto", sans-serif;
            }
            QPushButton {
                background-color: #3a86ff;
                color: white;
                border: none;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                border-radius: 30px;
                min-width: 200px;
            }
            QPushButton:hover {
                background-color: #2666cc;
            }
            QPushButton:pressed {
                background-color: #1a4c99;
            }
            QLabel {
                color: white;
                font-size: 18px;
            }
            QProgressDialog {
                background-color: #2a2a3c;
                color: white;
                border-radius: 10px;
            }
            QProgressBar {
                border-radius: 8px;
                text-align: center;
                background-color: #3a3a4e;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3a86ff;
                border-radius: 8px;
            }
        """)

    def mostrar_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = frame.shape
                bytes_per_line = ch * w
                qt_image = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(qt_image)
                self.video_label.setPixmap(pixmap.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            else:
                self.timer.stop()
                self.cap.release()

    def iniciar_analisis(self):
        archivo, _ = QFileDialog.getOpenFileName(self, "Seleccionar video", "",
                                         "Video Files (*.mp4 *.avi *.mov *.mkv)")
        if not archivo:
            return

        # Reproducir video con OpenCV
        if self.cap is not None:
            self.cap.release()
        self.cap = cv2.VideoCapture(archivo)
        if self.timer is None:
            self.timer = QTimer()
            self.timer.timeout.connect(self.mostrar_frame)
        self.timer.start(30)

        # Mostrar y resetear barra de progreso
        self.progress_bar.show()
        self.progress_bar.setValue(0)

        # Hilo y worker
        self.hilo = QThread()
        self.worker = AnalizadorWorker(archivo, self.model, self.device)
        self.worker.moveToThread(self.hilo)

        self.hilo.started.connect(self.worker.ejecutar)
        self.worker.terminado.connect(self.mostrar_resultado)
        self.worker.progreso.connect(self.progress_bar.setValue)   
        self.worker.terminado.connect(self.hilo.quit)
        self.worker.terminado.connect(self.worker.deleteLater)
        self.hilo.finished.connect(self.hilo.deleteLater)

        self.hilo.start()
        self.resultado_label.setText("Analizando...")
        self.resultado_label.setStyleSheet("font-size: 18px; padding: 10px; background-color: #2a2a3c; color: white; border: none;")


    def mostrar_resultado(self, texto):
        self.resultado_label.setText(texto)
        self.progress_bar.hide()  # ocultar barra al terminar
        
        # Extraer porcentaje si existe
        import re
        match = re.search(r'\(([\d\.]+)%\)', texto)
        
        if "Correcta" in texto:
            color = "#2ecc71"     
            bg_color = "#1e4a2f"
            estilo = f"""
                font-size: 22px;
                font-weight: bold;
                padding: 15px;
                border-radius: 15px;
                background-color: {bg_color};
                color: {color};
                border: 1px solid {color};
            """
        elif "riesgo" in texto or "Lesión" in texto:
            color = "#e74c3c"    
            bg_color = "#4a1a1a"
            estilo = f"""
                font-size: 22px;
                font-weight: bold;
                padding: 15px;
                border-radius: 15px;
                background-color: {bg_color};
                color: {color};
                border: 1px solid {color};
            """
        else:
            estilo = """
                font-size: 22px;
                font-weight: bold;
                padding: 15px;
                border-radius: 15px;
                background-color: #2a2a3c;
                color: white;
            """
        self.resultado_label.setStyleSheet(estilo)

    def cancelar_analisis(self):
        if self.hilo and self.hilo.isRunning():
            self.hilo.quit()
            self.hilo.wait()
            self.resultado_label.setText("Análisis cancelado.")

    def closeEvent(self, event):
        if self.cap is not None:
            self.cap.release()
        if self.timer is not None:
            self.timer.stop()
        event.accept()

# ------------------ PUNTO DE ENTRADA ------------------
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    ruta_modelo = r"c:\Users\Foche15\Desktop\tesina\IA\mejor_modelo_tesina.pth"
    model = cargar_modelo(ruta_modelo, device)
    
    print(f"Modelo cargado correctamente en {device}")
    
    app = QApplication(sys.argv)
    # Establecer ícono de la aplicación
    icon_path = os.path.join(os.path.dirname(__file__), "gym.png")
    if os.path.exists(icon_path):
        app.setWindowIcon(QIcon(icon_path))
    else:
        print(f"Advertencia: No se encontró el ícono en {icon_path}")
    ventana = VentanaPrincipal(model, device)
    ventana.show()
    sys.exit(app.exec_())