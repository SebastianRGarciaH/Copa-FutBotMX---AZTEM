import os
import cv2

def generar_video_prueba():
    # 1. Obtener la ruta raíz del proyecto
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(current_dir)
    
    # 2. Definir rutas exactas basadas en tu estructura actual
    video_input = os.path.join(root_dir, 'data', 'partido.mp4')
    output_dir = os.path.join(root_dir, 'results', 'visualizations')
    video_output = os.path.join(output_dir, 'demo_final.mp4')
    
    # Verificar que el video de prueba exista
    if not os.path.exists(video_input):
        print(f"Error: No se encontró el video en {video_input}")
        return
        
    # 3. Cargar el video original
    cap = cv2.VideoCapture(video_input)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Usar un FPS por defecto si el video no lo reporta bien
    if fps == 0 or fps != fps: 
        fps = 30.0
        
    # Preparar el escritor para el nuevo video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output, fourcc, fps, (width, height))
    
    print(f"Leyendo video desde: {video_input}")
    print(f"Guardando prueba en: {video_output}")
    print("Procesando 5 segundos de video para la prueba...")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # 4. Dibujar overlay de prueba
        cv2.putText(frame, "PRUEBA DE RUTAS - FUTBOTMX", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 255), 3)
        cv2.putText(frame, f"Frame: {frame_idx}", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    
        out.write(frame)
        frame_idx += 1
        
        # Limitar a 150 frames para que la prueba sea rápida
        if frame_idx >= 150:
            break
            
    cap.release()
    out.release()
    print("¡Prueba exitosa! Revisa la carpeta results/visualizations/ para ver tu archivo.")

if __name__ == "__main__":
    generar_video_prueba()