import sys
from PyQt6.QtWidgets import QApplication, QWidget, QLabel
from PyQt6.QtCore import Qt

class OverlayWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # --- CONFIGURACIÓN CLAVE DE LA VENTANA ---

        # 1. Hacer que la ventana no tenga bordes, barra de título, etc.
        self.setWindowFlag(Qt.WindowType.FramelessWindowHint)

        # 2. Hacer que la ventana siempre esté por encima de las demás.
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint)

        # 3. Hacer que el fondo de la ventana sea transparente.
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)

        # Hacemos que la ventana ocupe toda la pantalla.
        # Esto nos dará un "lienzo" invisible sobre todo el escritorio.
        self.showFullScreen()

        # --- AÑADIR CONTENIDO DE PRUEBA ---
        # Para verificar que el overlay funciona, añadimos una etiqueta de texto.
        # Si no pusiéramos nada, no veríamos la ventana transparente.
        
        # Creamos una etiqueta
        self.test_label = QLabel("Este es el overlay. Pulsa ESC para salir.", self)
        
        # Le damos un estilo para que sea visible
        self.test_label.setStyleSheet("""
            background-color: rgba(0, 0, 0, 150); 
            color: white; 
            font-size: 24px; 
            padding: 10px;
            border-radius: 5px;
        """)
        
        # La movemos a una posición en la pantalla (ej: 100px desde la izquierda, 100px desde arriba)
        self.test_label.move(100, 100)
        self.test_label.adjustSize() # Ajusta el tamaño de la etiqueta a su contenido

    # Añadimos una forma de cerrar la aplicación (ej: pulsando la tecla ESC)
    def keyPressEvent(self, event):
        if event.key() == Qt.Key.Key_Escape:
            self.close()

# --- BLOQUE PRINCIPAL PARA EJECUTAR LA APP ---
if __name__ == '__main__':
    app = QApplication(sys.argv)
    overlay = OverlayWindow()
    overlay.show()
    sys.exit(app.exec())