"""Utilidades comunes para proyectos de CV."""

import cv2


def read_video(path: str):
    """Generador que yield frames de un video."""
    cap = cv2.VideoCapture(path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        yield frame
    cap.release()


def write_video(frames: list, path: str, fps: float = 30.0):
    """Escribe una lista de frames a un archivo de video."""
    if not frames:
        return
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(path, fourcc, fps, (w, h))
    for frame in frames:
        writer.write(frame)
    writer.release()
