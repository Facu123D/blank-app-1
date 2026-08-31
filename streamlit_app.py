# -*- coding: utf-8 -*-
"""
Bar Path Tracker - Video Export con ImageIO FFmpeg (Compatible Web & Móvil)
"""

import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
import tempfile
import imageio
from scipy.signal import savgol_filter
from streamlit_image_coordinates import streamlit_image_coordinates

st.set_page_config(page_title="Bar Path Tracker", page_icon="🏋️‍♂️", layout="centered")

st.title("🏋️‍♂️ Bar Path Tracker")
st.write("Sube tu levantamiento, selecciona la barra en la foto y analiza el recorrido.")

video_file = st.file_uploader("1. Sube el video del levantamiento", type=["mp4", "mov", "avi"])

if video_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tfile.write(video_file.read())
    video_path = tfile.name

    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or np.isnan(fps):
        fps = 30.0
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()

    if ret:
        target_w = min(720, orig_w)
        # Asegurar dimensiones pares para FFmpeg H.264
        if target_w % 2 != 0: target_w -= 1
        scale_factor = target_w / float(orig_w)
        target_h = int(orig_h * scale_factor)
        if target_h % 2 != 0: target_h -= 1

        first_frame_resized = cv2.resize(first_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
        first_frame_rgb = cv2.cvtColor(first_frame_resized, cv2.COLOR_BGR2RGB)

        st.subheader("2. Toca la BARRA en la foto:")
        coords = streamlit_image_coordinates(first_frame_rgb, key="bar_coords")

        if coords is not None:
            cx, cy = int(coords["x"]), int(coords["y"])

            # Feedback visual de selección
            preview_img = first_frame_rgb.copy()
            box_size = int(max(24, min(50, target_h * 0.045)))
            half = box_size // 2

            cv2.line(preview_img, (cx, 0), (cx, target_h), (250, 204, 21), 2)
            cv2.rectangle(
                preview_img,
                (max(0, cx - half), max(0, cy - half)),
                (min(target_w, cx + half), min(target_h, cy + half)),
                (56, 189, 248), 2
            )
            cv2.circle(preview_img, (cx, cy), 6, (16, 185, 129), -1)

            st.image(preview_img, caption=f"🎯 Punto fijado: X={cx}px, Y={cy}px (Línea amarilla = Plomada)")

            if st.button("🚀 3. Analizar Movimiento", type="primary", use_container_width=True):
                with st.spinner("Procesando trayectoria con OpenCV y codificando video..."):
                    cap = cv2.VideoCapture(video_path)

                    if hasattr(cv2, 'TrackerCSRT_create'):
                        tracker = cv2.TrackerCSRT_create()
                    elif hasattr(cv2, 'legacy') and hasattr(cv2.legacy, 'TrackerCSRT_create'):
                        tracker = cv2.legacy.TrackerCSRT_create()
                    else:
                        tracker = cv2.TrackerKCF_create()

                    init_x = max(0, min(target_w - box_size, cx - half))
                    init_y = max(0, min(target_h - box_size, cy - half))

                    ret_f, f_frame = cap.read()
                    f_frame_res = cv2.resize(f_frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                    tracker.init(f_frame_res, (init_x, init_y, box_size, box_size))

                    frames_buffer = [f_frame_res]
                    raw_positions = [(cx, cy)]
                    plumb_x = cx

                    while cap.isOpened():
                        ret, frame = cap.read()
                        if not ret:
                            break
                        f_res = cv2.resize(frame, (target_w, target_h), interpolation=cv2.INTER_AREA)
                        frames_buffer.append(f_res)
                        success, bbox = tracker.update(f_res)
                        if success:
                            bx, by, bw, bh = [int(v) for v in bbox]
                            raw_positions.append((int(bx + bw / 2), int(by + bh / 2)))
                        else:
                            raw_positions.append(raw_positions[-1])

                    cap.release()
                    total_frames = len(frames_buffer)
                    dt = 1.0 / fps

                    times = np.array([i * dt for i in range(total_frames)])
                    y_coords = np.array([p[1] for p in raw_positions], dtype=np.float64)
                    x_coords = np.array([p[0] for p in raw_positions], dtype=np.float64)

                    if total_frames > 9:
                        w_len = min(13, total_frames if total_frames % 2 != 0 else total_frames - 1)
                        smooth_y = savgol_filter(y_coords, window_length=w_len, polyorder=2) if w_len >= 5 else y_coords
                        smooth_x = savgol_filter(x_coords, window_length=w_len, polyorder=2) if w_len >= 5 else x_coords
                    else:
                        smooth_y = np.convolve(y_coords, np.ones(5)/5, mode='same')
                        smooth_x = np.convolve(x_coords, np.ones(5)/5, mode='same')

                    pixels_per_cm = (target_h * 0.55) / 100.0

                    distancia_vertical_cm = (np.max(smooth_y) - np.min(smooth_y)) / pixels_per_cm
                    desvio_max_cm = np.max(np.abs(smooth_x - plumb_x)) / pixels_per_cm

                    dy_px = -np.gradient(smooth_y)
                    dy_m = (dy_px / pixels_per_cm) / 100.0
                    vertical_velocities = np.clip(dy_m / dt, 0.0, 3.5)
                    peak_vel = float(np.max(vertical_velocities))
                    concentric_pts = vertical_velocities[vertical_velocities > 0.05]
                    mean_vel = float(np.mean(concentric_pts)) if len(concentric_pts) > 0 else 0.0

                    # Renderizar cuadros y guardar con FFmpeg nativo
                    out_frames = []

                    for idx in range(total_frames):
                        frame = frames_buffer[idx].copy()
                        cv2.line(frame, (plumb_x, 0), (plumb_x, target_h), (250, 204, 21), 1, cv2.LINE_AA)

                        if idx > 1:
                            for k in range(1, idx + 1):
                                pt1 = (int(smooth_x[k - 1]), int(smooth_y[k - 1]))
                                pt2 = (int(smooth_x[k]), int(smooth_y[k]))
                                v = vertical_velocities[k]
                                hue = int(np.clip(120 - (v * 90), 0, 120))
                                color_bgr = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
                                bgr = (int(color_bgr[0]), int(color_bgr[1]), int(color_bgr[2]))
                                cv2.line(frame, pt1, pt2, bgr, 3, cv2.LINE_AA)

                        curr_cx, curr_cy = int(smooth_x[idx]), int(smooth_y[idx])
                        cv2.circle(frame, (curr_cx, curr_cy), 5, (255, 255, 255), -1, cv2.LINE_AA)
                        cv2.circle(frame, (curr_cx, curr_cy), 2, (16, 185, 129), -1, cv2.LINE_AA)

                        # HUD métricas
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (15, 15), (320, 110), (15, 23, 42), -1)
                        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)
                        cv2.rectangle(frame, (15, 15), (320, 110), (51, 65, 85), 1, cv2.LINE_AA)

                        cv2.putText(frame, "BAR PATH TRACKER", (25, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (52, 211, 153), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Desvio: {desvio_max_cm:.1f} cm", (25, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)
                        cv2.putText(frame, f"Recorrido: {distancia_vertical_cm:.1f} cm", (25, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (56, 189, 248), 2, cv2.LINE_AA)
                        cv2.putText(frame, f"Vel. actual: {vertical_velocities[idx]:.2f} m/s", (25, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (226, 232, 240), 1, cv2.LINE_AA)

                        # Convertir a RGB para ImageIO
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        out_frames.append(frame_rgb)

                    temp_dir = tempfile.gettempdir()
                    out_video_path = os.path.join(temp_dir, "out_video.mp4")

                    # Exportación H.264 compatible universalmente
                    imageio.mimwrite(
                        out_video_path,
                        out_frames,
                        fps=fps,
                        codec='libx264',
                        quality=8,
                        pixelformat='yuv420p'
                    )

                    # Gráfico de velocidad
                    fig, ax = plt.subplots(figsize=(8, 3.8), dpi=140)
                    plt.style.use('dark_background')
                    fig.patch.set_facecolor('#0B0F17')
                    ax.set_facecolor('#111827')
                    ax.plot(times, vertical_velocities, color='#10b981', linewidth=2.5, label='Velocidad Subida (m/s)')
                    ax.fill_between(times, vertical_velocities, color='#10b981', alpha=0.25)
                    ax.axhline(mean_vel, color='#06b6d4', linestyle='--', linewidth=1.5, label=f'Vel. Media: {mean_vel:.2f} m/s')
                    peak_idx = int(np.argmax(vertical_velocities))
                    ax.scatter([times[peak_idx]], [peak_vel], color='#f43f5e', s=70, zorder=6, label=f'Vel. Pico: {peak_vel:.2f} m/s')
                    ax.set_title(f"Curva de Velocidad (ROM: {distancia_vertical_cm:.1f} cm | Desvío: {desvio_max_cm:.1f} cm)", fontsize=10, fontweight='bold', pad=10, color='#F1F5F9')
                    ax.set_xlabel("Tiempo (segundos)", fontsize=8.5, color='#94A3B8')
                    ax.set_ylabel("Velocidad (m/s)", fontsize=8.5, color='#94A3B8')
                    ax.grid(True, linestyle=':', alpha=0.3, color='#475569')
                    ax.legend(loc='upper right', fontsize=8, framealpha=0.85, facecolor='#0F172A', edgecolor='#334155')

                    st.markdown("### 📊 Resultados:")
                    col1, col2, col3, col4 = st.columns(4)
                    col1.metric("Distancia Total", f"{distancia_vertical_cm:.1f} cm")
                    col2.metric("Desvío Lateral", f"{desvio_max_cm:.1f} cm")
                    col3.metric("Velocidad Pico", f"{peak_vel:.2f} m/s")
                    col4.metric("Velocidad Media", f"{mean_vel:.2f} m/s")

                    st.pyplot(fig)

                    st.subheader("🎥 Video con Trazo:")
                    with open(out_video_path, 'rb') as vf:
                        video_bytes = vf.read()
                        st.video(video_bytes, format="video/mp4")
