import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

class PushupDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Inicializar Mediapipe Pose.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        # Contador y estado.
        self.counter = 0
        self.stage = None
        # Inicializar video.
        self.cap = cv2.VideoCapture(0)
        # Umbrales de ángulo para las flexiones.
        self.angle_thresholds = {'up': 90, 'down': 160}
        # Para el cálculo de la velocidad.
        self.previous_time = None
        self.previous_elbow_angle = None
        self.speeds = deque(maxlen=20)  # Guardar últimas velocidades para promediar.
        # Para graficar.
        self.graph_points = deque(maxlen=100)
        # Fuente para textos.
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    def calculate_angle(self, a, b, c):
        """
        Calcula el ángulo en el punto 'b' formado por los segmentos 'ab' y 'bc'.
        """
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)

        # Vectorización de los segmentos.
        ba = a - b
        bc = c - b

        # Cálculo del ángulo.
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)
        angle = np.arccos(cosine_angle)
        angle = np.degrees(angle)

        return angle

    def calculate_speed(self, current_angle, current_time):
        """
        Calcula la velocidad angular en grados por segundo.
        """
        if self.previous_time is None or self.previous_elbow_angle is None:
            speed = 0
        else:
            delta_angle = abs(current_angle - self.previous_elbow_angle)
            delta_time = current_time - self.previous_time
            if delta_time > 0:
                speed = delta_angle / delta_time
            else:
                speed = 0
        # Actualizar valores previos.
        self.previous_time = current_time
        self.previous_elbow_angle = current_angle
        return speed

    def run(self):
        with self.pose as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("No se puede leer desde la cámara. Saliendo...")
                    break

                # Voltear el frame horizontalmente para efecto espejo.
                frame = cv2.flip(frame, 1)
                frame_copy = frame.copy()

                # Aplicar filtros y detectar movimiento.
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blurred = cv2.GaussianBlur(gray, (5, 5), 0)
                # Detección de movimiento simple usando diferencia de frames.
                if hasattr(self, 'previous_frame'):
                    frame_diff = cv2.absdiff(self.previous_frame, blurred)
                    _, thresh = cv2.threshold(frame_diff, 25, 255, cv2.THRESH_BINARY)
                    # Mostrar detección de movimiento.
                    cv2.imshow('Movimiento', thresh)
                self.previous_frame = blurred

                # Procesamiento de Mediapipe.
                image_rgb = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # Extraer landmarks.
                if results.pose_landmarks:
                    try:
                        landmarks = results.pose_landmarks.landmark
                        image_height, image_width, _ = frame.shape

                        # Obtener coordenadas necesarias.
                        shoulder = landmarks[self.mp_pose.PoseLandmark.LEFT_SHOULDER.value]
                        elbow = landmarks[self.mp_pose.PoseLandmark.LEFT_ELBOW.value]
                        wrist = landmarks[self.mp_pose.PoseLandmark.LEFT_WRIST.value]
                        hip = landmarks[self.mp_pose.PoseLandmark.LEFT_HIP.value]

                        # Convertir a coordenadas en píxeles.
                        shoulder_coords = [int(shoulder.x * image_width), int(shoulder.y * image_height)]
                        elbow_coords = [int(elbow.x * image_width), int(elbow.y * image_height)]
                        wrist_coords = [int(wrist.x * image_width), int(wrist.y * image_height)]
                        hip_coords = [int(hip.x * image_width), int(hip.y * image_height)]

                        # Calcular el ángulo en el codo.
                        angle_elbow = self.calculate_angle(shoulder_coords, elbow_coords, wrist_coords)
                        # Calcular el ángulo en el hombro.
                        angle_shoulder = self.calculate_angle(elbow_coords, shoulder_coords, hip_coords)

                        # Calcular velocidad angular.
                        current_time = time.time()
                        speed = self.calculate_speed(angle_elbow, current_time)
                        self.speeds.append(speed)
                        avg_speed = np.mean(self.speeds)

                        # Agregar punto para graficar.
                        self.graph_points.append(angle_elbow)

                        # Visualizar los ángulos y velocidad.
                        cv2.putText(frame, f'Elbow Angle: {int(angle_elbow)}', (10, 100),
                                    self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'Speed: {avg_speed:.2f} deg/s', (10, 130),
                                    self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        # Estimación de calorías (simplificada).
                        calories_burned = self.counter * 0.5  # Supongamos 0.5 calorías por flexión.
                        cv2.putText(frame, f'Calories Burned: {calories_burned:.1f}', (10, 160),
                                    self.font, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

                        # Detección de flexiones.
                        if angle_elbow > self.angle_thresholds['down'] and angle_shoulder > self.angle_thresholds['down']:
                            self.stage = "down"
                        elif angle_elbow < self.angle_thresholds['up'] and angle_shoulder < self.angle_thresholds['up'] and self.stage == 'down':
                            self.stage = "up"
                            self.counter += 1
                            print(f"Conteo de flexiones: {self.counter}")

                        # Dibujar puntos clave y líneas.
                        # Puntos del brazo izquierdo.
                        cv2.circle(frame, tuple(shoulder_coords), 5, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, tuple(elbow_coords), 5, (0, 0, 255), cv2.FILLED)
                        cv2.circle(frame, tuple(wrist_coords), 5, (0, 0, 255), cv2.FILLED)
                        # Puntos de la cadera.
                        cv2.circle(frame, tuple(hip_coords), 5, (0, 0, 255), cv2.FILLED)

                        # Líneas del brazo y torso.
                        cv2.line(frame, tuple(shoulder_coords), tuple(elbow_coords), (0, 0, 255), 2)
                        cv2.line(frame, tuple(elbow_coords), tuple(wrist_coords), (0, 0, 255), 2)
                        cv2.line(frame, tuple(shoulder_coords), tuple(hip_coords), (0, 0, 255), 2)

                    except Exception as e:
                        print(f"Error en la detección: {e}")

                # Mostrar contador y estado.
                self.display_counter(frame)

                # Graficar ángulo del codo.
                self.plot_graph(frame)

                # Mostrar imagen.
                cv2.imshow('Detector de Flexiones', frame)

                # Salir con la tecla 'q'.
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Liberar recursos.
            self.cap.release()
            cv2.destroyAllWindows()

    def display_counter(self, frame):
        """
        Muestra el contador y el estado en la imagen.
        """
        # Fondo del contador.
        cv2.rectangle(frame, (0, 0), (225, 75), (245, 117, 16), -1)
        # Texto del contador.
        cv2.putText(frame, 'REPS', (15, 20),
                    self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    self.font, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Texto del estado.
        cv2.putText(frame, 'STAGE', (100, 20),
                    self.font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.stage if self.stage else '', (100, 60),
                    self.font, 2, (255, 255, 255), 2, cv2.LINE_AA)

    def plot_graph(self, frame):
        """
        Dibuja un gráfico del ángulo del codo en el tiempo.
        """
        graph_height = 200
        graph_width = 400
        graph = np.zeros((graph_height, graph_width, 3), dtype=np.uint8)

        if len(self.graph_points) > 1:
            # Escalar puntos al tamaño del gráfico.
            scaled_points = np.interp(self.graph_points, [0, 180], [graph_height, 0])
            points = np.array([[i * (graph_width / len(self.graph_points)), y] for i, y in enumerate(scaled_points)], dtype=np.int32)
            # Dibujar líneas entre los puntos.
            cv2.polylines(graph, [points], False, (0, 255, 0), 2)

        # Colocar el gráfico en la esquina inferior derecha del frame.
        x_offset = frame.shape[1] - graph_width
        y_offset = frame.shape[0] - graph_height
        frame[y_offset:y_offset + graph_height, x_offset:x_offset + graph_width] = graph

if __name__ == "__main__":
    detector = PushupDetector()
    detector.run()
