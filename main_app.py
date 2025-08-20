import sys
import threading
from PyQt6.QtWidgets import QApplication, QWidget, QLabel, QPushButton, QHBoxLayout, QStyle, QFrame
from PyQt6.QtCore import Qt, pyqtSignal
from pynput import mouse
import mss
import easyocr
import numpy as np
import cv2
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# --- PARÁMETROS PARA EXPERIMENTAR (Basados en tu sugerencia) ---
CAPTURE_WIDTH = 500
CAPTURE_HEIGHT = 300
MOUSE_IDLE_TIME = 0.7
LINE_TOLERANCE = 8      # Más estricto verticalmente
HORIZONTAL_TOLERANCE_PIXELS = 25 # Gap horizontal máximo en píxeles

# -----------------------------------------------------------------------------
# CLASE 1: EL OVERLAY DE PANTALLA COMPLETA (CON LÓGICA REFINADA)
# -----------------------------------------------------------------------------
class MainApp(QWidget):
    update_translation_signal = pyqtSignal(list)

    def __init__(self):
        super().__init__()
        self.current_mode = "pointer" 
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
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        self.update_translation_signal.connect(self.update_translation_labels)

    def start_mouse_listener(self):
        self.mouse_listener = mouse.Listener(on_move=self.on_mouse_move)
        self.mouse_listener.start()

    def on_mouse_move(self, x, y):
        if self.current_mode != "pointer": return
        if self.ocr_timer: self.ocr_timer.cancel()
        self.update_translation_signal.emit([])
        self.ocr_timer = threading.Timer(MOUSE_IDLE_TIME, self.perform_ocr_thread)
        self.ocr_timer.start()
        
    def perform_ocr_thread(self):
        threading.Thread(target=self.perform_ocr, daemon=True).start()
    
    # --- TU LÓGICA DE AGRUPACIÓN CONSERVADORA ---
    def group_text_fragments_by_line_conservative(self, fragments):
        if not fragments: return []
        fragments.sort(key=lambda f: (f[0][0][1], f[0][0][0]))
        
        lines = []
        if not fragments: return lines
        
        lines.append([fragments[0]]) # Empezar con la primera línea
        
        for i in range(1, len(fragments)):
            curr_fragment = fragments[i]
            curr_box = curr_fragment[0]
            
            # Intentar añadir a una línea existente
            added_to_existing = False
            for line in lines:
                last_fragment_in_line = line[-1]
                last_box_in_line = last_fragment_in_line[0]
                
                y_diff = abs(curr_box[0][1] - last_box_in_line[0][1])
                x_gap = curr_box[0][0] - last_box_in_line[1][0]
                
                if y_diff < LINE_TOLERANCE and 0 < x_gap < HORIZONTAL_TOLERANCE_PIXELS:
                    line.append(curr_fragment)
                    added_to_existing = True
                    break
            
            if not added_to_existing:
                # Si no encajó en ninguna línea existente, es una nueva línea
                lines.append([curr_fragment])
        
        # Procesar las líneas para devolver el formato correcto
        processed_lines = []
        for line in lines:
            line.sort(key=lambda f: f[0][0][0])
            full_text = " ".join([f[1] for f in line])
            start_bbox = line[0][0]
            processed_lines.append({'bbox': start_bbox, 'text': full_text})
            
        return processed_lines
        
    def perform_ocr(self):
        pos_x, pos_y = self.mouse_controller.position
        bounding_box = { 'top': int(pos_y - CAPTURE_HEIGHT / 2), 'left': int(pos_x - CAPTURE_WIDTH / 2), 'width': CAPTURE_WIDTH, 'height': CAPTURE_HEIGHT }
        with mss.mss() as sct:
            sct_img = sct.grab(bounding_box)
            img_np = np.array(sct_img)
            img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
            raw_results = self.reader.readtext(img_gray)
            
            # Usando el nuevo algoritmo de agrupación conservador
            grouped_lines = self.group_text_fragments_by_line_conservative(raw_results)
            
            translations_to_show = []
            if grouped_lines:
                print("\n--- Analizando y Traduciendo (Modo Conservador) ---")
                for line in grouped_lines:
                    text, bbox = line['text'], line['bbox']
                    try:
                        detected_lang = detect(text)
                        if detected_lang != 'es':
                            translated_text = self.translator.translate(text)
                            if translated_text and translated_text.lower() != text.lower():
                                print(f'  "{text}" -> "{translated_text}"')
                                top_left = bbox[0]
                                abs_x, abs_y = bounding_box['left'] + top_left[0], bounding_box['top'] + top_left[1]
                                final_x, final_y = int(abs_x / self.screen_scale_factor), int(abs_y / self.screen_scale_factor)
                                translations_to_show.append({'text': translated_text, 'pos': (final_x, final_y)})
                    except Exception:
                        pass
                self.update_translation_signal.emit(translations_to_show)

    # --- TU NUEVA FUNCIÓN PARA CREAR LABELS COMPACTOS ---
    def create_compact_translation_label(self, text, position):
        label = QLabel(text, self)
        label.setStyleSheet("""
            QLabel {
                background-color: rgba(20, 20, 20, 230);
                color: #FFFFFF;
                border: 1px solid rgba(255, 255, 255, 80);
                border-radius: 4px;
                padding: 3px 6px;
                font-family: 'Segoe UI', Arial, sans-serif;
                font-size: 13px;
                font-weight: normal;
            }
        """)
        label.setWordWrap(False) # No permitir que el texto salte de línea
        label.move(position[0], position[1])
        label.adjustSize()
        
        # Truncar si es demasiado largo para evitar labels gigantes
        if label.width() > 350:
            truncated_text = text[:40] + "..." if len(text) > 40 else text
            label.setText(truncated_text)
            label.adjustSize()
            
        label.show()
        return label

    def update_translation_labels(self, translations):
        for label in self.translation_labels: label.deleteLater()
        self.translation_labels.clear()
        for item in translations:
            if not item['text']: continue
            # Usando la nueva función para crear los labels
            new_label = self.create_compact_translation_label(item['text'], item['pos'])
            self.translation_labels.append(new_label)

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape: self.close()

    def closeEvent(self, event):
        print("Cerrando aplicación...")
        if self.ocr_timer: self.ocr_timer.cancel()
        if hasattr(self, 'mouse_listener'): self.mouse_listener.stop()
        if hasattr(self, 'toolbar'): self.toolbar.close()
        event.accept()

# -----------------------------------------------------------------------------
# CLASE 2: LA BARRA DE HERRAMIENTAS FLOTANTE (Sin cambios)
# -----------------------------------------------------------------------------
class Toolbar(QWidget):
    # ... (El código de la clase Toolbar es el mismo que en la versión anterior)
    mode_changed = pyqtSignal(str)
    clear_labels_signal = pyqtSignal()
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Traductor")
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)
        self.drag_pos, self.is_minimized, self.active_button = None, False, None
        self.main_layout = QHBoxLayout(self); self.main_layout.setContentsMargins(0,0,0,0)
        self.expanded_widget = QWidget(); layout = QHBoxLayout(self.expanded_widget)
        self.btn_pointer = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), ""); self.btn_pointer.setToolTip("Modo Puntero (Automático)"); self.btn_pointer.clicked.connect(lambda: self.set_active_mode("pointer", self.btn_pointer)); layout.addWidget(self.btn_pointer)
        self.btn_fullscreen = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon), ""); self.btn_fullscreen.setToolTip("Traducir Pantalla Completa"); self.btn_fullscreen.clicked.connect(lambda: self.set_active_mode("fullscreen", self.btn_fullscreen)); layout.addWidget(self.btn_fullscreen)
        self.btn_rect_select = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), ""); self.btn_rect_select.setToolTip("Seleccionar Área Rectangular"); self.btn_rect_select.clicked.connect(lambda: self.set_active_mode("rect_select", self.btn_rect_select)); layout.addWidget(self.btn_rect_select)
        self.btn_lasso_select = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_CustomBase), ""); self.btn_lasso_select.setToolTip("Seleccionar con Lazo"); self.btn_lasso_select.clicked.connect(lambda: self.set_active_mode("lasso", self.btn_lasso_select)); layout.addWidget(self.btn_lasso_select)
        separator = QFrame(); separator.setFrameShape(QFrame.Shape.VLine); separator.setFrameShadow(QFrame.Shadow.Sunken); layout.addWidget(separator)
        self.btn_settings = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView), ""); self.btn_settings.setToolTip("Configurar Idiomas"); self.btn_settings.clicked.connect(self.on_settings_clicked); layout.addWidget(self.btn_settings)
        self.btn_collapse = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarShadeButton), ""); self.btn_collapse.setToolTip("Minimizar barra de herramientas"); self.btn_collapse.clicked.connect(self.toggle_minimize); layout.addWidget(self.btn_collapse)
        self.btn_trash = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), ""); self.btn_trash.setToolTip("Limpiar traducciones actuales"); self.btn_trash.clicked.connect(self.on_trash_clicked); layout.addWidget(self.btn_trash)
        self.minimized_widget = QWidget(); min_layout = QHBoxLayout(self.minimized_widget)
        self.btn_expand = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft), ""); self.btn_expand.setToolTip("Expandir barra de herramientas"); self.btn_expand.clicked.connect(self.toggle_minimize); min_layout.addWidget(self.btn_expand)
        self.main_layout.addWidget(self.expanded_widget); self.main_layout.addWidget(self.minimized_widget); self.minimized_widget.hide()
        self.setStyleSheet("""QWidget { background-color: #2E2E2E; border-radius: 8px; } QPushButton { background-color: #4A4A4A; border: 1px solid #6E6E6E; padding: 6px; border-radius: 5px; } QPushButton:hover { background-color: #5A5A5A; } QPushButton:pressed { background-color: #6A6A6A; } QPushButton[active="true"] { background-color: #3A3A3A; border: 2px solid #0078D7; border-style: inset; } QFrame { border: 1px solid #4A4A4A; }""")
        self.set_active_mode("pointer", self.btn_pointer)
    def toggle_minimize(self): self.is_minimized = not self.is_minimized; self.expanded_widget.setHidden(self.is_minimized); self.minimized_widget.setHidden(not self.is_minimized); self.setFixedSize(self.sizeHint())
    def set_active_mode(self, mode_name, button_widget):
        if self.active_button: self.active_button.setProperty("active", "false"); self.active_button.style().unpolish(self.active_button); self.active_button.style().polish(self.active_button)
        button_widget.setProperty("active", "true"); button_widget.style().unpolish(button_widget); button_widget.style().polish(button_widget)
        self.active_button = button_widget; self.mode_changed.emit(mode_name)
    def on_settings_clicked(self): print("Acción: Abrir Configuración")
    def on_trash_clicked(self): self.clear_labels_signal.emit()
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton: self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft(); event.accept()
    def mouseMoveEvent(self, event):
        if event.buttons() == Qt.MouseButton.LeftButton and self.drag_pos: self.move(event.globalPosition().toPoint() - self.drag_pos); event.accept()
        
# -----------------------------------------------------------------------------
# BLOQUE DE EJECUCIÓN PRINCIPAL
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    app = QApplication(sys.argv)
    overlay = MainApp()
    toolbar = Toolbar()
    overlay.toolbar = toolbar
    def handle_mode_change(mode): overlay.current_mode = mode; overlay.update_translation_signal.emit([])
    toolbar.mode_changed.connect(handle_mode_change)
    toolbar.clear_labels_signal.connect(lambda: overlay.update_translation_signal.emit([]))
    overlay.show()
    toolbar.show()
    toolbar.setFixedSize(toolbar.sizeHint())
    sys.exit(app.exec())