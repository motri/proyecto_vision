import cv2
import mediapipe as mp
import numpy as np

class FlexionDetector:
    def __init__(self, min_detection_confidence=0.7, min_tracking_confidence=0.7):
        # Inicializar Mediapipe Pose.
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(min_detection_confidence=min_detection_confidence,
                                      min_tracking_confidence=min_tracking_confidence)
        # Contadores y estados para ambos brazos.
        self.counters = {'left': 0, 'right': 0}
        self.stages = {'left': None, 'right': None}
        # Inicializar video.
        self.cap = cv2.VideoCapture(0)
        # Definir los umbrales de ángulo.
        self.angle_thresholds = {'up': 30, 'down': 160}

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

    def run(self):
        with self.pose as pose:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("No se puede leer desde la cámara. Saliendo...")
                    break

                # Voltear y convertir el frame.
                frame = cv2.flip(frame, 1)
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Procesamiento de Mediapipe.
                results = pose.process(image_rgb)

                # Extraer landmarks.
                if results.pose_landmarks:
                    try:
                        landmarks = results.pose_landmarks.landmark
                        image_height, image_width, _ = frame.shape

                        # Procesar ambos brazos.
                        for side in ['left', 'right']:
                            shoulder = landmarks[getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_SHOULDER').value]
                            elbow = landmarks[getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_ELBOW').value]
                            wrist = landmarks[getattr(self.mp_pose.PoseLandmark, f'{side.upper()}_WRIST').value]

                            # Convertir a coordenadas en píxeles.
                            shoulder_coords = [int(shoulder.x * image_width), int(shoulder.y * image_height)]
                            elbow_coords = [int(elbow.x * image_width), int(elbow.y * image_height)]
                            wrist_coords = [int(wrist.x * image_width), int(wrist.y * image_height)]

                            # Calcular el ángulo.
                            angle = self.calculate_angle(shoulder_coords, elbow_coords, wrist_coords)

                            # Visualizar el ángulo.
                            cv2.putText(frame, f'{side.capitalize()} Angle: {int(angle)}',
                                        (elbow_coords[0] - 50, elbow_coords[1] - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)

                            # Detección de flexiones.
                            if angle > self.angle_thresholds['down']:
                                self.stages[side] = "down"
                            elif angle < self.angle_thresholds['up'] and self.stages[side] == 'down':
                                self.stages[side] = "up"
                                self.counters[side] += 1
                                print(f"Conteo de flexiones ({side}): {self.counters[side]}")

                            # Dibujar puntos clave y líneas.
                            cv2.circle(frame, tuple(shoulder_coords), 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, tuple(elbow_coords), 5, (0, 255, 0), cv2.FILLED)
                            cv2.circle(frame, tuple(wrist_coords), 5, (0, 255, 0), cv2.FILLED)
                            cv2.line(frame, tuple(shoulder_coords), tuple(elbow_coords), (0, 255, 0), 2)
                            cv2.line(frame, tuple(elbow_coords), tuple(wrist_coords), (0, 255, 0), 2)

                            # Mostrar contador y estado.
                            self.display_counter(frame, side)

                    except Exception as e:
                        print(f"Error en la detección: {e}")

                # Mostrar imagen.
                cv2.imshow('Detector de Flexiones', frame)

                # Salir con la tecla 'q'.
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

            # Liberar recursos.
            self.cap.release()
            cv2.destroyAllWindows()

    def display_counter(self, frame, side):
        """
        Muestra el contador y el estado en la imagen.
        """
        x_offset = 10 if side == 'left' else frame.shape[1] - 230
        # Fondo del contador.
        cv2.rectangle(frame, (x_offset, 0), (x_offset + 220, 75), (245, 117, 16), -1)
        # Texto del contador.
        cv2.putText(frame, f'{side.capitalize()} REPS', (x_offset + 10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, str(self.counters[side]), (x_offset + 10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)
        # Texto del estado.
        cv2.putText(frame, 'STAGE', (x_offset + 100, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(frame, self.stages[side] if self.stages[side] else '', (x_offset + 100, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

if __name__ == "__main__":
    detector = FlexionDetector()
    detector.run()
