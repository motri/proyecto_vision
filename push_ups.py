import cv2
import mediapipe as mp
import numpy as np

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

    def apply_filters(self, frame):
        """
        Aplica filtros al frame para mejorar la detección de contornos.
        """
        # Convertir a escala de grises.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Aplicar desenfoque Gaussiano.
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Aplicar detección de bordes con Canny.
        edges = cv2.Canny(blurred, 50, 150)
        # Aplicar dilatación y erosión para cerrar huecos en los bordes.
        kernel = np.ones((5,5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        edges = cv2.erode(edges, kernel, iterations=1)
        return edges

    def detect_contours(self, edges):
        """
        Detecta contornos en la imagen filtrada.
        """
        # Encontrar contornos en la imagen de bordes.
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

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

                # Aplicar filtros y detectar contornos.
                filtered_frame = self.apply_filters(frame)
                contours = self.detect_contours(filtered_frame)

                # Dibujar contornos detectados en el frame original.
                cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

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

                        # Visualizar los ángulos.
                        cv2.putText(frame, f'Elbow: {int(angle_elbow)}', (elbow_coords[0] - 50, elbow_coords[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
                        cv2.putText(frame, f'Shoulder: {int(angle_shoulder)}', (shoulder_coords[0] - 70, shoulder_coords[1] - 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

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
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counter), (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Texto del estado.
        cv2.putText(frame, 'STAGE', (100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.stage if self.stage else '', (100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    detector = PushupDetector()
    detector.run()
