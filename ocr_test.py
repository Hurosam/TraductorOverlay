import mss
import easyocr
import numpy as np
import cv2 # OpenCV

print("Inicializando el lector de EasyOCR (puede tardar la primera vez)...")
# 1. Inicializamos el lector de OCR.
# Le decimos que busque texto en inglés ('en'). Puedes añadir más idiomas como ['en', 'es'].
reader = easyocr.Reader(['en'], gpu=False) # Usamos CPU para mayor compatibilidad
print("EasyOCR listo.")

# 2. Definimos el área de la pantalla que queremos capturar.
# Es un diccionario con las coordenadas (en píxeles) desde la esquina superior izquierda.
# ¡AJUSTA ESTOS VALORES PARA QUE CAPTUREN ALGÚN TEXTO EN TU PANTALLA!
bounding_box = {'top': 100, 'left': 100, 'width': 400, 'height': 200}

# 3. Usamos MSS para capturar la pantalla.
with mss.mss() as sct:
    # Capturamos la imagen del área definida
    sct_img = sct.grab(bounding_box)

    # 4. Convertimos la imagen capturada a un formato que EasyOCR pueda leer.
    # mss la devuelve en formato BGRA, la convertimos a RGB usando numpy y OpenCV.
    img_np = np.array(sct_img)
    img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)

    print("\nImagen capturada. Realizando reconocimiento de texto...")

    # 5. Pasamos la imagen al lector de OCR.
    results = reader.readtext(img_rgb)

    # 6. Imprimimos los resultados.
    if not results:
        print("No se encontró texto en el área seleccionada.")
    else:
        print("\n--- Texto encontrado: ---")
        for (bbox, text, prob) in results:
            # `bbox` son las coordenadas del texto dentro de la captura
            # `text` es el texto extraído
            # `prob` es la confianza de la detección (de 0 a 1)
            print(f'Texto: "{text}" (Confianza: {prob:.2f})')
        print("------------------------")