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
from langdetect import detect, LangDetectException

# --- CONFIGURACIÓN ---
CAPTURE_WIDTH = 500
CAPTURE_HEIGHT = 300
MOUSE_IDLE_TIME = 0.7
# --- NUEVO: Tolerancia para agrupar texto en la misma línea (en píxeles) ---
LINE_TOLERANCE_PIXELS = 10

class MainApp(QWidget):
    update_translation_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        print("Cargando modelo de EasyOCR (en/es)...")
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
        print("¡Modelo cargado!")
        self.translator = GoogleTranslator(source='auto', target='es')
        self.translation_labels = []
        self.mouse_controller = mouse.Controller()
        self.ocr_timer = None
        app = QApplication.instance()
        self.screen_scale_factor = app.primaryScreen().devicePixelRatio()
        print(f"Factor de escala de pantalla detectado: {self.screen_scale_factor}")
        self.initUI()
        self.start_mouse_listener()

    def initUI(self):
        # ... (sin cambios)
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        self.update_translation_signal.connect(self.update_translation_labels)

    def start_mouse_listener(self):
        # ... (sin cambios)
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()

    def on_mouse_move(self, x, y):
        # ... (sin cambios)
        if self.ocr_timer:
            self.ocr_timer.cancel()
        self.update_translation_signal.emit([])
        self.ocr_timer = threading.Timer(MOUSE_IDLE_TIME, self.perform_ocr_thread)
        self.ocr_timer.start()
        
    def perform_ocr_thread(self):
        # ... (sin cambios)
        threading.Thread(target=self.perform_ocr, daemon=True).start()
    
    # --- NUEVA FUNCIÓN: El cerebro de la agrupación ---
    def group_text_fragments_by_line(self, fragments):
        if not fragments:
            return []
        
        # Ordenar fragmentos de arriba a abajo, luego de izquierda a derecha
        fragments.sort(key=lambda f: (f[0][0][1], f[0][0][0]))
        
        lines = []
        current_line = [fragments[0]]
        
        for i in range(1, len(fragments)):
            prev_box = current_line[-1][0]
            curr_box = fragments[i][0]
            
            # Calcular el centro vertical de las cajas
            prev_center_y = (prev_box[0][1] + prev_box[2][1]) / 2
            curr_center_y = (curr_box[0][1] + curr_box[2][1]) / 2
            
            # Si están verticalmente alineados, pertenecen a la misma línea
            if abs(prev_center_y - curr_center_y) < LINE_TOLERANCE_PIXELS:
                current_line.append(fragments[i])
            else:
                # Si no, la línea anterior ha terminado. La guardamos y empezamos una nueva.
                lines.append(current_line)
                current_line = [fragments[i]]
        
        lines.append(current_line) # Añadir la última línea
        
        # Ahora, para cada línea, unimos el texto
        processed_lines = []
        for line in lines:
            line.sort(key=lambda f: f[0][0][0]) # Ordenar por X para el texto final
            full_text = " ".join([f[1] for f in line])
            start_bbox = line[0][0] # Coordenadas de la primera palabra
            processed_lines.append({'bbox': start_bbox, 'text': full_text})
            
        return processed_lines

    def perform_ocr(self):
        pos_x, pos_y = self.mouse_controller.position
        bounding_box = { 'top': int(pos_y - CAPTURE_HEIGHT / 2), 'left': int(pos_x - CAPTURE_WIDTH / 2), 'width': CAPTURE_WIDTH, 'height': CAPTURE_HEIGHT }
        
        with mss.mss() as sct:
            sct_img = sct.grab(bounding_box)
            img_np = np.array(sct_img)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
            
            # 1. Obtenemos los fragmentos troceados del OCR
            raw_results = self.reader.readtext(img_gray)
            
            # 2. Los pasamos a nuestra nueva función de agrupación
            grouped_lines = self.group_text_fragments_by_line(raw_results)
            
            translations_to_show = []
            if grouped_lines:
                print("\n--- Analizando Líneas Agrupadas ---")
                # 3. Iteramos sobre las LÍNEAS, no sobre los fragmentos
                for line in grouped_lines:
                    text = line['text']
                    bbox = line['bbox']
                    try:
                        detected_lang = detect(text)
                        if detected_lang != 'es':
                            print(f'Línea: "{text}" (Idioma: {detected_lang}). Traduciendo...')
                            translated_text = self.translator.translate(text)
                            if translated_text is None: continue

                            top_left = bbox[0]
                            abs_x = bounding_box['left'] + top_left[0]
                            abs_y = bounding_box['top'] + top_left[1]
                            final_x = int(abs_x / self.screen_scale_factor)
                            final_y = int(abs_y / self.screen_scale_factor)
                            
                            translations_to_show.append({'text': translated_text, 'pos': (final_x, final_y)})
                        else:
                            print(f'Línea: "{text}" (Idioma: {detected_lang}). Saltando.')
                    except Exception as e:
                        # Manejo de errores simplificado
                        print(f"No se pudo procesar la línea '{text}': {e}")
                
                self.update_translation_signal.emit(translations_to_show)

    def update_translation_labels(self, translations):
        # ... (sin cambios)
        for label in self.translation_labels:
            label.deleteLater()
        self.translation_labels.clear()
        for item in translations:
            if not item['text']: continue
            label = QLabel(item['text'], self)
            label.setStyleSheet("""
                background-color: rgba(20, 20, 20, 230); color: #f0f0f0; 
                font-size: 15px; font-family: Segoe UI; padding: 4px;
                border-radius: 4px; border: 1px solid rgba(255, 255, 255, 50);
            """)
            label.move(item['pos'][0], item['pos'][1])
            label.adjustSize()
            label.show()
            self.translation_labels.append(label)

    def keyPressEvent(self, event):
        # ... (sin cambios)
        if event.key() == Qt.Key.Key_Escape:
            self.close()

    def closeEvent(self, event):
        # ... (sin cambios)
        print("Cerrando aplicación...")
        if self.ocr_timer:
            self.ocr_timer.cancel()
        self.mouse_listener.stop()
        event.accept()

if __name__ == '__main__':
    # ... (sin cambios)
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec())