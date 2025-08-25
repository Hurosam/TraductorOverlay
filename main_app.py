import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from PyQt6.QtWidgets import (QApplication, QWidget, QLabel, QPushButton, 
                             QHBoxLayout, QStyle, QFrame, QGraphicsDropShadowEffect)
from PyQt6.QtCore import Qt, pyqtSignal, QRect, QTimer
from PyQt6.QtGui import QPainter, QPen, QColor, QBrush, QGuiApplication
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
LINE_TOLERANCE = 10
HORIZONTAL_TOLERANCE_PIXELS = 30
RECT_REFRESH_INTERVAL = 2000 # 2 segundos

# -----------------------------------------------------------------------------
# CLASE 1: APLICACIÓN PRINCIPAL (OVERLAY)
# -----------------------------------------------------------------------------
class MainApp(QWidget):
    add_label_signal = pyqtSignal(str, dict)
    remove_label_signal = pyqtSignal(str)
    clear_all_labels_signal = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.current_mode = "pointer"
        self.is_selecting = False
        self.selection_rect = QRect()
        self.completed_selection_rect = QRect()
        self.mouse_listener = None
        self.mouse_controller = mouse.Controller()

        print("Cargando modelo de EasyOCR (en/es)...")
        self.reader = easyocr.Reader(['en', 'es'], gpu=False)
        print("¡Modelo cargado!")
        self.translator = GoogleTranslator(source='auto', target='es')
        
        self.active_translations = {}
        self.ocr_timer = None
        self.screen_scale_factor = QGuiApplication.primaryScreen().devicePixelRatio()
        
        self.fullscreen_timer = QTimer(self); self.fullscreen_timer.setSingleShot(True)
        self.fullscreen_timer.timeout.connect(self.perform_fullscreen_ocr)
        
        self.rect_refresh_timer = QTimer(self)
        self.rect_refresh_timer.timeout.connect(self.trigger_periodic_selection_ocr)

        self.initUI()
        self.set_mode("pointer")

    def initUI(self):
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.showFullScreen()
        self.add_label_signal.connect(self.add_label_slot)
        self.remove_label_signal.connect(self.remove_label_slot)
        self.clear_all_labels_signal.connect(self.clear_all_labels_slot)
        self.setMouseTracking(True)

    def set_mode(self, mode):
        print(f"Cambiando a modo: {mode}")
        previous_mode = self.current_mode
        self.current_mode = mode
        self.is_selecting = False
        self.selection_rect = QRect()

        if self.ocr_timer: self.ocr_timer.cancel()
        self.fullscreen_timer.stop(); self.rect_refresh_timer.stop()
        
        if not (previous_mode == "rect_select" and mode == "pointer"):
            self.completed_selection_rect = QRect()
        
        self.clear_all_labels_signal.emit()

        if mode == "pointer":
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setCursor(Qt.CursorShape.ArrowCursor); self.start_pointer_listener()
            if previous_mode == "rect_select" and self.completed_selection_rect.isValid():
                self.rect_refresh_timer.start(RECT_REFRESH_INTERVAL)
        elif mode == "fullscreen":
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
            self.setCursor(Qt.CursorShape.ArrowCursor); self.stop_pointer_listener()
            self.fullscreen_timer.start(250)
        elif mode == "rect_select":
            self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, False)
            self.setCursor(Qt.CursorShape.CrossCursor); self.stop_pointer_listener()
        
        self.update()

    def mousePressEvent(self, event):
        if self.current_mode == "rect_select" and event.button() == Qt.MouseButton.LeftButton:
            self.is_selecting = True
            self.selection_rect.setTopLeft(event.pos()); self.selection_rect.setBottomRight(event.pos())
            self.completed_selection_rect = QRect(); self.clear_all_labels_signal.emit()
            self.update()

    def mouseMoveEvent(self, event):
        if self.is_selecting and self.current_mode == "rect_select":
            self.selection_rect.setBottomRight(event.pos()); self.update()

    def mouseReleaseEvent(self, event):
        if self.current_mode == "rect_select" and self.is_selecting:
            self.is_selecting = False
            if self.selection_rect.width() > 10 and self.selection_rect.height() > 10:
                self.completed_selection_rect = self.selection_rect.normalized()
                self.set_mode("pointer"); self.trigger_periodic_selection_ocr()
            else:
                self.set_mode("pointer")
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self); painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        if self.current_mode == "rect_select":
            overlay_color = QColor(0, 0, 0, 70); painter.fillRect(self.rect(), overlay_color)
            rect_to_draw = self.selection_rect.normalized() if self.is_selecting else self.completed_selection_rect
            if not rect_to_draw.isNull():
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear); painter.fillRect(rect_to_draw, Qt.BrushStyle.SolidPattern)
                painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
                pen = QPen(); pen.setWidth(2); pen.setColor(QColor(50, 150, 255) if self.is_selecting else QColor(80, 220, 100))
                pen.setStyle(Qt.PenStyle.SolidLine if self.is_selecting else Qt.PenStyle.DashLine)
                painter.setPen(pen); painter.drawRect(rect_to_draw)
        elif self.current_mode == "pointer" and self.completed_selection_rect.isValid():
            pen = QPen(QColor(80, 220, 100), 2, Qt.PenStyle.DashLine); painter.setPen(pen); painter.drawRect(self.completed_selection_rect)

    def trigger_periodic_selection_ocr(self):
        if not self.completed_selection_rect.isValid(): self.rect_refresh_timer.stop(); return
        threading.Thread(target=self.perform_ocr_on_selection, daemon=True).start()
    
    def perform_ocr_on_selection(self):
        try:
            rect = self.completed_selection_rect
            bbox = {'top': int(rect.y() * self.screen_scale_factor), 'left': int(rect.x() * self.screen_scale_factor),
                    'width': int(rect.width() * self.screen_scale_factor), 'height': int(rect.height() * self.screen_scale_factor)}
            # --- CORRECCIÓN CRÍTICA: El origen debe estar en píxeles físicos ---
            origin = (bbox['left'], bbox['top'])
            self.run_ocr_and_update(bbox, origin, is_periodic=True)
        except Exception as e: print(f"Error en hilo de OCR de selección: {e}")
        
    def perform_fullscreen_ocr(self):
        print("\n--- Iniciando OCR de pantalla completa ---")
        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                origin = (monitor['left'], monitor['top'])
                threading.Thread(target=self.run_ocr_and_update, args=(monitor, origin, False), daemon=True).start()
        except Exception as e: print(f"Error en OCR de pantalla completa: {e}")

    # --- CAMBIO CRÍTICO: Lógica de traducción paralela ---
    def run_ocr_and_update(self, bounding_box, selection_origin_physical, is_periodic):
        if not is_periodic: self.clear_all_labels_signal.emit()
        
        new_results = {}
        with mss.mss() as sct:
            sct_img = sct.grab(bounding_box)
            img_np = np.array(sct_img); img_gray = cv2.cvtColor(img_np, cv2.COLOR_BGRA2GRAY)
            raw_results = self.reader.readtext(img_gray, width_ths=0.6, text_threshold=0.6, low_text=0.4)
            grouped_lines = self.group_text_fragments_by_line(raw_results)
            
            items_to_translate = []
            for line in grouped_lines:
                text, bbox_coords = line['text'], line['bbox']
                try:
                    detected_lang = detect(text)
                    if detected_lang != 'es': items_to_translate.append(line)
                except (LangDetectException, TypeError): continue
            
            # Función que traduce un solo item
            def translate_item(item):
                try:
                    translated_text = self.translator.translate(item['text'])
                    if translated_text and translated_text.lower() != item['text'].lower():
                        return item['text'], {'translated': translated_text, 'bbox': item['bbox'], 'selection_origin': selection_origin_physical}
                except Exception: return None
                return None

            # Usamos un ThreadPool para traducir todo en paralelo y mostrarlo a medida que llega
            with ThreadPoolExecutor() as executor:
                # En modo periódico, actualizamos al final para evitar parpadeo
                if is_periodic:
                    future_results = [executor.submit(translate_item, item) for item in items_to_translate]
                    for future in future_results:
                        result = future.result()
                        if result: new_results[result[0]] = result[1]
                    self.update_active_translations(new_results)
                # En modo "1 por 1", actualizamos en cuanto llega el resultado
                else:
                    for item in items_to_translate:
                        future = executor.submit(translate_item, item)
                        future.add_done_callback(lambda fut: self.add_label_signal.emit(fut.result()[0], fut.result()[1]) if fut.result() else None)

    def update_active_translations(self, new_results):
        old_keys = set(self.active_translations.keys()); new_keys = set(new_results.keys())
        for key in old_keys - new_keys: self.remove_label_signal.emit(key)
        for key in new_keys - old_keys: self.add_label_signal.emit(key, new_results[key])

    def start_pointer_listener(self):
        if not self.mouse_listener: self.mouse_listener = mouse.Listener(on_move=self.on_pointer_move); self.mouse_listener.start()

    def stop_pointer_listener(self):
        if self.mouse_listener: self.mouse_listener.stop(); self.mouse_listener = None
        if self.ocr_timer: self.ocr_timer.cancel(); self.ocr_timer = None

    def on_pointer_move(self, x, y):
        if self.current_mode != "pointer" or self.rect_refresh_timer.isActive(): return
        if self.ocr_timer: self.ocr_timer.cancel()
        self.clear_all_labels_signal.emit()
        self.ocr_timer = threading.Timer(MOUSE_IDLE_TIME, self.perform_ocr_pointer); self.ocr_timer.start()

    def perform_ocr_pointer(self):
        if self.current_mode != "pointer": return
        try:
            pos_x, pos_y = self.mouse_controller.position
            bbox = {'top': int(pos_y - CAPTURE_HEIGHT / 2), 'left': int(pos_x - CAPTURE_WIDTH / 2), 'width': CAPTURE_WIDTH, 'height': CAPTURE_HEIGHT}
            origin = (bbox['left'], bbox['top'])
            self.run_ocr_and_update(bbox, origin, is_periodic=False)
        except Exception as e: print(f"Error en OCR puntero: {e}")
        
    def group_text_fragments_by_line(self, fragments):
        if not fragments: return []
        fragments = [f for f in fragments if f[2] > 0.3]; fragments.sort(key=lambda f: (f[0][0][1], f[0][0][0]))
        if not fragments: return []
        lines = []; current_line = [fragments[0]]
        for i in range(1, len(fragments)):
            prev_box = current_line[-1][0]; curr_box = fragments[i][0]
            y_diff = abs(curr_box[0][1] - prev_box[0][1]); x_gap = curr_box[0][0] - prev_box[1][0]
            if y_diff < LINE_TOLERANCE and 0 < x_gap < HORIZONTAL_TOLERANCE_PIXELS: current_line.append(fragments[i])
            else: lines.append(current_line); current_line = [fragments[i]]
        lines.append(current_line)
        processed_lines = []
        for line_fragments in lines:
            full_text = " ".join([f[1] for f in line_fragments])
            if len(full_text.strip()) < 2 or full_text.strip().isdigit(): continue
            min_x = line_fragments[0][0][0][0]; max_x = line_fragments[-1][0][1][0]
            min_y = min(f[0][0][1] for f in line_fragments); max_y = max(f[0][2][1] for f in line_fragments)
            full_bbox = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
            processed_lines.append({'bbox': full_bbox, 'text': full_text})
        return processed_lines

    # --- CAMBIO CRÍTICO: Lógica de tamaño flexible ---
    def create_compact_translation_label(self, text, bbox, selection_origin_physical):
        label = QLabel(text, self); label.setAlignment(Qt.AlignmentFlag.AlignCenter); label.setWordWrap(True)
        label.setStyleSheet("""QLabel{background-color:rgb(40,40,40);color:#F0F0F0;border:1px solid rgba(255,255,255,60);border-radius:3px;padding:3px 5px;font-family:'Segoe UI',Arial,sans-serif;font-size:13px;font-weight:500}""")
        top_left = bbox[0]; bottom_right = bbox[2]
        
        final_x = (selection_origin_physical[0] + top_left[0]) / self.screen_scale_factor
        final_y = (selection_origin_physical[1] + top_left[1]) / self.screen_scale_factor
        final_w = (bottom_right[0] - top_left[0]) / self.screen_scale_factor
        
        # El ancho es fijo, pero la altura es flexible
        label.setMaximumWidth(int(final_w))
        label.adjustSize() # Ajusta la altura al contenido
        label.move(int(final_x), int(final_y))
        label.show()
        return label

    def add_label_slot(self, original_text, data):
        if original_text in self.active_translations: return
        new_label = self.create_compact_translation_label(data['translated'], data['bbox'], data['selection_origin'])
        self.active_translations[original_text] = new_label

    def remove_label_slot(self, original_text):
        if original_text in self.active_translations:
            label = self.active_translations.pop(original_text); label.deleteLater()

    def clear_all_labels_slot(self):
        for key in list(self.active_translations.keys()):
            label = self.active_translations.pop(key); label.deleteLater()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape: self.close()

    def closeEvent(self, event):
        print("Cerrando aplicación..."); self.stop_pointer_listener()
        self.fullscreen_timer.stop(); self.rect_refresh_timer.stop()
        if self.ocr_timer: self.ocr_timer.cancel()
        if hasattr(self, 'toolbar'): self.toolbar.close()
        event.accept()

# -----------------------------------------------------------------------------
# CLASE 2: LA BARRA DE HERRAMIENTAS FLOTANTE (VERSIÓN COMPLETA)
# -----------------------------------------------------------------------------
class Toolbar(QWidget):
    mode_changed = pyqtSignal(str)
    clear_labels_signal = pyqtSignal()
    
    def __init__(self):
        super().__init__(); self.setWindowTitle("Traductor"); self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.drag_pos = None; self.is_minimized = False; self.active_button = None
        self.main_layout = QHBoxLayout(self); self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.expanded_widget = QWidget(); layout = QHBoxLayout(self.expanded_widget)
        self.btn_pointer = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowUp), ""); self.btn_pointer.setToolTip("Modo Puntero (Automático)"); self.btn_pointer.clicked.connect(lambda: self.set_active_mode("pointer", self.btn_pointer)); layout.addWidget(self.btn_pointer)
        self.btn_fullscreen = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_DesktopIcon), ""); self.btn_fullscreen.setToolTip("Traducir Pantalla Completa"); self.btn_fullscreen.clicked.connect(lambda: self.set_active_mode("fullscreen", self.btn_fullscreen)); layout.addWidget(self.btn_fullscreen)
        self.btn_rect_select = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogDetailedView), ""); self.btn_rect_select.setToolTip("Seleccionar Área Rectangular"); self.btn_rect_select.clicked.connect(lambda: self.set_active_mode("rect_select", self.btn_rect_select)); layout.addWidget(self.btn_rect_select)
        separator = QFrame(); separator.setFrameShape(QFrame.Shape.VLine); separator.setFrameShadow(QFrame.Shadow.Sunken); layout.addWidget(separator)
        self.btn_settings = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_FileDialogInfoView), ""); self.btn_settings.setToolTip("Configurar Idiomas"); self.btn_settings.clicked.connect(self.on_settings_clicked); layout.addWidget(self.btn_settings)
        self.btn_collapse = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TitleBarShadeButton), ""); self.btn_collapse.setToolTip("Minimizar barra de herramientas"); self.btn_collapse.clicked.connect(self.toggle_minimize); layout.addWidget(self.btn_collapse)
        self.btn_trash = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_TrashIcon), ""); self.btn_trash.setToolTip("Limpiar traducciones actuales"); self.btn_trash.clicked.connect(self.on_trash_clicked); layout.addWidget(self.btn_trash)
        self.minimized_widget = QWidget(); min_layout = QHBoxLayout(self.minimized_widget)
        self.btn_expand = QPushButton(self.style().standardIcon(QStyle.StandardPixmap.SP_ArrowLeft), ""); self.btn_expand.setToolTip("Expandir barra de herramientas"); self.btn_expand.clicked.connect(self.toggle_minimize); min_layout.addWidget(self.btn_expand)
        self.main_layout.addWidget(self.expanded_widget); self.main_layout.addWidget(self.minimized_widget); self.minimized_widget.hide()
        self.setStyleSheet("""QWidget{background-color:#2E2E2E;border-radius:8px}QPushButton{background-color:#4A4A4A;border:1px solid #6E6E6E;padding:5px;border-radius:5px;min-width:28px;min-height:28px}QPushButton:hover{background-color:#5A5A5A}QPushButton:pressed{background-color:#6A6A6A}QPushButton[active="true"]{background-color:#0078D7;border:2px solid #40A0FF}QFrame{border:1px solid #4A4A4A}""")
        self.set_active_mode("pointer", self.btn_pointer)

    def toggle_minimize(self):
        self.is_minimized = not self.is_minimized; self.expanded_widget.setHidden(self.is_minimized); self.minimized_widget.setHidden(not self.is_minimized)
        self.setFixedSize(self.sizeHint())

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
    toolbar.mode_changed.connect(overlay.set_mode)
    toolbar.clear_labels_signal.connect(overlay.clear_all_labels_slot)
    overlay.show(); toolbar.show(); toolbar.setFixedSize(toolbar.sizeHint())
    sys.exit(app.exec())