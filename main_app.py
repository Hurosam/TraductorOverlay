import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt, pyqtSignal
from pynput import mouse
import mss
import easyocr
import numpy as np
import cv2
from deep_translator import GoogleTranslator

# --- CONFIGURACIÓN ---
CAPTURE_WIDTH = 500
CAPTURE_HEIGHT = 300
MOUSE_IDLE_TIME = 0.7

class MainApp(QWidget):
    update_translation_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        
        # --- MEJORA 1: Soporte para múltiples idiomas en OCR ---
        print("Cargando modelo de EasyOCR (en/es)...")
        # Le decimos que busque tanto en inglés como en español
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
        print("¡Modelo cargado!")

        self.translator = GoogleTranslator(source='auto', target='es')
        
        self.translation_labels = []
        self.mouse_controller = mouse.Controller()
        self.ocr_timer = None

        # --- MEJORA 2: Detectar el factor de escala de la pantalla ---
        app = QApplication.instance()
        self.screen_scale_factor = app.primaryScreen().devicePixelRatio()
        print(f"Factor de escala de pantalla detectado: {self.screen_scale_factor}")
        
        self.initUI()
        self.start_mouse_listener()

    def initUI(self):
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        self.update_translation_signal.connect(self.update_translation_labels)

    def start_mouse_listener(self):
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()

    def on_mouse_move(self, x, y):
        if self.ocr_timer:
            self.ocr_timer.cancel()
        self.update_translation_signal.emit([])
        self.ocr_timer = threading.Timer(MOUSE_IDLE_TIME, self.perform_ocr_thread)
        self.ocr_timer.start()
        
    def perform_ocr_thread(self):
        threading.Thread(target=self.perform_ocr, daemon=True).start()
        
    def perform_ocr(self):
        pos_x, pos_y = self.mouse_controller.position
        
        bounding_box = {
            'top': int(pos_y - CAPTURE_HEIGHT / 2),
            'left': int(pos_x - CAPTURE_WIDTH / 2),
            'width': CAPTURE_WIDTH,
            'height': CAPTURE_HEIGHT,
        }
        
        with mss.mss() as sct:
            sct_img = sct.grab(bounding_box)
            img_np = np.array(sct_img)

            # --- MEJORA 3: Pre-procesamiento de la imagen a escala de grises ---
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
            
            # Pasamos la imagen pre-procesada al OCR
            results = self.reader.readtext(img_gray)
            
            translations_to_show = []
            if results:
                print("\n--- Detectado y Traduciendo ---")
                for (bbox, text, prob) in results:
                    if prob < 0.4: # Ignoramos detecciones con muy baja confianza
                        continue
                    try:
                        translated_text = self.translator.translate(text)
                        if translated_text is None: continue

                        print(f'"{text}" (Conf: {prob:.2f}) -> "{translated_text}"')
                        
                        top_left = bbox[0]
                        abs_x = bounding_box['left'] + top_left[0]
                        abs_y = bounding_box['top'] + top_left[1]

                        # --- MEJORA 4: Corregimos las coordenadas con el factor de escala ---
                        final_x = int(abs_x / self.screen_scale_factor)
                        final_y = int(abs_y / self.screen_scale_factor)
                        
                        translations_to_show.append({
                            'text': translated_text,
                            'pos': (final_x, final_y)
                        })
                    except Exception as e:
                        print(f"Error en la traducción de '{text}': {e}")
                
                self.update_translation_signal.emit(translations_to_show)

    def update_translation_labels(self, translations):
        for label in self.translation_labels:
            label.deleteLater()
        self.translation_labels.clear()

        for item in translations:
            if not item['text']: continue
            
            label = QLabel(item['text'], self)
            label.setStyleSheet("""
                background-color: rgba(20, 20, 20, 230); 
                color: #f0f0f0; 
                font-size: 15px; 
                font-family: Segoe UI;
                padding: 4px;
                border-radius: 4px;
                border: 1px solid rgba(255, 255, 255, 50);
            """)
            label.move(item['pos'][0], item['pos'][1])
            label.adjustSize()
            label.show()
            self.translation_labels.append(label)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        print("Cerrando aplicación...")
        if self.ocr_timer:
            self.ocr_timer.cancel()
        self.mouse_listener.stop()
        event.accept()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())