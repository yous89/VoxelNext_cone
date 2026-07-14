import argparse, time
from pathlib import Path
import cv2
import numpy as np
import torch
import ouster.sdk as sdk
import ouster.sdk.core as core

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.models import build_network, load_data_to_gpu
from pcdet.utils import common_utils


class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, logger):
        super().__init__(dataset_cfg, class_names, False, Path("."), logger)

    def prepare_one_frame(self, points, frame_id):
        data_dict = self.prepare_data({"points": points, "frame_id": frame_id})
        return self.collate_batch([data_dict])


def get_first_scan(scan_item):
    if hasattr(scan_item, "__len__") and hasattr(scan_item, "__getitem__"):
        return scan_item[0]
    return scan_item


def scan_to_points(scan, xyz_lut):
    xyz = xyz_lut(scan).reshape(-1, 3).astype(np.float32)

    reflectivity = scan.field(
        core.ChanField.REFLECTIVITY
    ).reshape(-1, 1).astype(np.float32)

    reflectivity = reflectivity / 10.0

    points = np.concatenate([xyz, reflectivity], axis=1)

    mask = np.isfinite(points).all(axis=1)
    mask &= np.linalg.norm(points[:, :3], axis=1) > 0.1

    return points[mask]

def world_to_pixel(x, y, x_min, x_max, y_min, y_max, w, h):
    px = int((x - x_min) / (x_max - x_min) * w)
    py = int(h - (y - y_min) / (y_max - y_min) * h)
    return px, py



def count_points_in_rotated_box(points, box):
    """
    Zählt die LiDAR-Punkte innerhalb einer gedrehten 3D-Bounding-Box.

    box:
        x, y, z, dx, dy, dz, yaw
    """
    if len(points) == 0:
        return 0

    x, y, z, dx, dy, dz, yaw = box[:7]

    relative = points[:, :3].copy()
    relative[:, 0] -= x
    relative[:, 1] -= y
    relative[:, 2] -= z

    cos_yaw = np.cos(-yaw)
    sin_yaw = np.sin(-yaw)

    local_x = (
        relative[:, 0] * cos_yaw
        - relative[:, 1] * sin_yaw
    )
    local_y = (
        relative[:, 0] * sin_yaw
        + relative[:, 1] * cos_yaw
    )
    local_z = relative[:, 2]

    inside = (
        (np.abs(local_x) <= dx / 2.0) &
        (np.abs(local_y) <= dy / 2.0) &
        (np.abs(local_z) <= dz / 2.0)
    )

    return int(np.count_nonzero(inside))


def draw_bev(points, boxes, scores, frame_id, model_fps, pipeline_fps, args):
    # Dunkler Hintergrund
    img = np.ones(
        (args.height, args.width, 3),
        dtype=np.uint8
    ) * 24

    x_min = args.x_min
    x_max = args.x_max
    y_min = args.y_min
    y_max = args.y_max

    # Punktwolke auf den sichtbaren Bereich begrenzen
    visible_mask = (
        (points[:, 0] >= x_min) &
        (points[:, 0] <= x_max) &
        (points[:, 1] >= y_min) &
        (points[:, 1] <= y_max)
    )

    visible_points = points[visible_mask]

    # LiDAR-Punkte farbig nach Entfernung darstellen
    if len(visible_points) > 0:
        distances = np.linalg.norm(
            visible_points[:, :2],
            axis=1
        )

        d_min = float(distances.min())
        d_max = float(distances.max())

        normalized = (
            (distances - d_min) /
            (d_max - d_min + 1e-6)
        )

        color_values = np.clip(
            normalized * 255,
            0,
            255
        ).astype(np.uint8)

        colors = cv2.applyColorMap(
            color_values,
            cv2.COLORMAP_PLASMA
        ).reshape(-1, 3)

        px = (
            (visible_points[:, 0] - x_min) /
            (x_max - x_min) *
            args.width
        ).astype(np.int32)

        py = (
            args.height -
            (visible_points[:, 1] - y_min) /
            (y_max - y_min) *
            args.height
        ).astype(np.int32)

        valid = (
            (px >= 0) &
            (px < args.width) &
            (py >= 0) &
            (py < args.height)
        )

        px = px[valid]
        py = py[valid]
        colors = colors[valid]

        # 1-Pixel-Punkte
        img[py, px] = colors


    # LiDAR-Ursprung
    ego_x, ego_y = world_to_pixel(
        0.0,
        0.0,
        x_min,
        x_max,
        y_min,
        y_max,
        args.width,
        args.height
    )

    if (
        0 <= ego_x < args.width and
        0 <= ego_y < args.height
    ):
        cv2.circle(
            img,
            (ego_x, ego_y),
            6,
            (255, 255, 255),
            2,
            lineType=cv2.LINE_AA
        )

        cv2.putText(
            img,
            "LiDAR",
            (ego_x + 10, ego_y + 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

    object_infos = []

    # Vorhersagen zeichnen
    for object_id, (box, score) in enumerate(
        zip(boxes, scores),
        start=1
    ):
        x, y, z, dx, dy, dz, yaw = box[:7]

        # Box-Eckpunkte im lokalen Koordinatensystem
        corners = np.array([
            [ dx / 2.0,  dy / 2.0],
            [ dx / 2.0, -dy / 2.0],
            [-dx / 2.0, -dy / 2.0],
            [-dx / 2.0,  dy / 2.0]
        ], dtype=np.float32)

        rotation = np.array([
            [np.cos(yaw), -np.sin(yaw)],
            [np.sin(yaw),  np.cos(yaw)]
        ], dtype=np.float32)

        corners = corners @ rotation.T
        corners[:, 0] += x
        corners[:, 1] += y

        pixel_corners = np.array([
            world_to_pixel(
                corner_x,
                corner_y,
                x_min,
                x_max,
                y_min,
                y_max,
                args.width,
                args.height
            )
            for corner_x, corner_y in corners
        ], dtype=np.int32)

        # Box nur zeichnen, wenn mindestens ein Teil sichtbar ist
        if (
            np.any(pixel_corners[:, 0] >= 0) and
            np.any(pixel_corners[:, 0] < args.width) and
            np.any(pixel_corners[:, 1] >= 0) and
            np.any(pixel_corners[:, 1] < args.height)
        ):
            cv2.polylines(
                img,
                [pixel_corners],
                True,
                (40, 255, 80),
                4,
                lineType=cv2.LINE_AA
            )

        center_x, center_y = world_to_pixel(
            x,
            y,
            x_min,
            x_max,
            y_min,
            y_max,
            args.width,
            args.height
        )

        point_count = count_points_in_rotated_box(
            points,
            box
        )

        object_infos.append({
            "id": object_id,
            "class": "small_cone",
            "points": point_count,
            "confidence": float(score)
        })

        # Label direkt neben der Box
        if (
            0 <= center_x < args.width and
            0 <= center_y < args.height
        ):
            label_lines = [
                f"ID {object_id} | small_cone",
                f"{point_count} pts | conf {score:.2f}"
            ]

            label_width = 185
            label_height = 46

            label_x = min(
                center_x + 8,
                args.width - label_width - 5
            )
            label_y = max(
                center_y - label_height - 8,
                5
            )

            overlay = img.copy()

            cv2.rectangle(
                overlay,
                (label_x, label_y),
                (
                    label_x + label_width,
                    label_y + label_height
                ),
                (5, 5, 5),
                -1
            )

            img = cv2.addWeighted(
                overlay,
                0.78,
                img,
                0.22,
                0
            )

            cv2.rectangle(
                img,
                (label_x, label_y),
                (
                    label_x + label_width,
                    label_y + label_height
                ),
                (40, 255, 80),
                1,
                lineType=cv2.LINE_AA
            )

            cv2.putText(
                img,
                label_lines[0],
                (label_x + 6, label_y + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )

            cv2.putText(
                img,
                label_lines[1],
                (label_x + 6, label_y + 37),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (40, 255, 80),
                1,
                cv2.LINE_AA
            )

    # HUD-Panel oben links
    panel_x1 = 12
    panel_y1 = 12
    panel_width = 430

    visible_rows = min(len(object_infos), 8)
    panel_height = 188 + visible_rows * 31

    panel_x2 = panel_x1 + panel_width
    panel_y2 = panel_y1 + panel_height

    overlay = img.copy()

    cv2.rectangle(
        overlay,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (5, 5, 5),
        -1
    )

    img = cv2.addWeighted(
        overlay,
        0.82,
        img,
        0.18,
        0
    )

    cv2.rectangle(
        img,
        (panel_x1, panel_y1),
        (panel_x2, panel_y2),
        (210, 210, 210),
        1,
        lineType=cv2.LINE_AA
    )

    # Hauptinformationen weiß
    cv2.putText(
        img,
        f"Frame: {frame_id}",
        (28, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"Model FPS: {model_fps:.2f}",
        (28, 78),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"Pipeline FPS: {pipeline_fps:.2f}",
        (28, 111),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        f"Boxes: {len(boxes)}",
        (28, 144),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.78,
        (255, 255, 255),
        2,
        cv2.LINE_AA
    )

    cv2.line(
        img,
        (25, 161),
        (panel_x2 - 15, 161),
        (180, 180, 180),
        1,
        cv2.LINE_AA
    )

    # Tabellenkopf
    cv2.putText(
        img,
        "ID",
        (28, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "Class",
        (70, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "Points",
        (205, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    cv2.putText(
        img,
        "Confidence",
        (285, 184),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.46,
        (220, 220, 220),
        1,
        cv2.LINE_AA
    )

    # Objektzeilen
    for row_index, info in enumerate(object_infos[:8]):
        row_y = 214 + row_index * 31

        cv2.putText(
            img,
            str(info["id"]),
            (30, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (40, 255, 80),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            info["class"],
            (70, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        cv2.putText(
            img,
            f'{info["points"]} pts',
            (205, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 255, 80),
            1,
            cv2.LINE_AA
        )

        confidence = info["confidence"]

        cv2.putText(
            img,
            f"{confidence:.2f}",
            (285, row_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (40, 255, 80),
            1,
            cv2.LINE_AA
        )

        # Confidence-Balken
        bar_x1 = 345
        bar_y1 = row_y - 12
        bar_width = 65
        bar_height = 8

        cv2.rectangle(
            img,
            (bar_x1, bar_y1),
            (bar_x1 + bar_width, bar_y1 + bar_height),
            (80, 80, 80),
            -1
        )

        filled_width = int(
            np.clip(confidence, 0.0, 1.0) *
            bar_width
        )

        cv2.rectangle(
            img,
            (bar_x1, bar_y1),
            (
                bar_x1 + filled_width,
                bar_y1 + bar_height
            ),
            (40, 255, 80),
            -1
        )

    if len(object_infos) > 8:
        cv2.putText(
            img,
            f"+ {len(object_infos) - 8} weitere Objekte",
            (28, panel_y2 - 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (220, 220, 220),
            1,
            cv2.LINE_AA
        )

    return img


def temporal_filter(boxes, scores, previous_boxes, max_lateral_distance=0.35):
    """
    Prüft ausschließlich die seitliche Position y.
    Die x-Position nach vorne wird vollständig ignoriert.
    """
    if previous_boxes is None or len(previous_boxes) == 0 or len(boxes) == 0:
        return (
            np.empty((0, boxes.shape[1]), dtype=boxes.dtype),
            np.empty((0,), dtype=scores.dtype)
        )

    previous_y = previous_boxes[:, 1]
    keep_indices = []

    for index, box in enumerate(boxes):
        current_y = box[1]

        # Nur links/rechts vergleichen, x wird nicht benutzt
        lateral_distances = np.abs(previous_y - current_y)

        if lateral_distances.min() <= max_lateral_distance:
            keep_indices.append(index)

    if len(keep_indices) == 0:
        return (
            np.empty((0, boxes.shape[1]), dtype=boxes.dtype),
            np.empty((0,), dtype=scores.dtype)
        )

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    return boxes[keep_indices], scores[keep_indices]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg_file", default="cfgs/kitti_models/voxelnext_cones.yaml")
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--pcap", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--score_thresh", type=float, default=0.9)
    parser.add_argument("--output", default="../demo_video/voxelnext_bev_clean.mp4")
    parser.add_argument("--max_frames", type=int, default=0)
    parser.add_argument("--width", type=int, default=1200)
    parser.add_argument("--height", type=int, default=900)
    parser.add_argument("--fps_video", type=int, default=10)

    parser.add_argument("--x_min", type=float, default=2.0)
    parser.add_argument("--x_max", type=float, default=16.0)
    parser.add_argument("--y_min", type=float, default=-8.0)
    parser.add_argument("--y_max", type=float, default=5.0)
    parser.add_argument("--show_scores", action="store_true")

    args = parser.parse_args()

    logger = common_utils.create_logger()
    cfg_from_yaml_file(args.cfg_file, cfg)

    # Weniger Top-K-Kandidaten, damit auch dünne Frames funktionieren
    cfg.MODEL.DENSE_HEAD.POST_PROCESSING.MAX_OBJ_PER_SAMPLE = 100
    cfg.MODEL.DENSE_HEAD.TARGET_ASSIGNER_CONFIG.NUM_MAX_OBJS = 100

    source = sdk.open_source(args.pcap, meta=[args.metadata], sensor_idx=0, collate=False)

    with open(args.metadata, "r") as f:
        sensor_info = core.SensorInfo(f.read())

    xyz_lut = core.XYZLut(sensor_info)
    dataset = DemoDataset(cfg.DATA_CONFIG, cfg.CLASS_NAMES, logger)

    model = build_network(cfg.MODEL, len(cfg.CLASS_NAMES), dataset)
    model.load_params_from_file(args.ckpt, logger=logger, to_cpu=False)
    model.cuda()
    model.eval()

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*"mp4v"),
                             args.fps_video, (args.width, args.height))

    previous_boxes = None

    with torch.no_grad():
        for frame_id, scan_item in enumerate(source):
            if args.max_frames > 0 and frame_id >= args.max_frames:
                break

            frame_start = time.time()
            scan = get_first_scan(scan_item)
            display_points = scan_to_points(scan, xyz_lut)

            # Für das Modell den bisher gut funktionierenden Bereich verwenden
            model_mask = (
                (display_points[:, 0] >= 0.0) &
                (display_points[:, 0] <= 35.0) &
                (display_points[:, 1] >= -35.0) &
                (display_points[:, 1] <= 35.0) &
                (display_points[:, 2] >= -2.0) &
                (display_points[:, 2] <= 1.0)
            )
            points = display_points[model_mask]

            if len(points) == 0:
                print(f"Frame {frame_id:04d} übersprungen: keine Modellpunkte.")
                continue

            data_dict = dataset.prepare_one_frame(points, frame_id)
            load_data_to_gpu(data_dict)

            torch.cuda.synchronize()
            inference_start = time.time()

            pred_dicts, _ = model.forward(data_dict)

            torch.cuda.synchronize()
            inference_time = time.time() - inference_start
            model_fps = 1.0 / max(inference_time, 1e-6)
            boxes = pred_dicts[0]["pred_boxes"].detach().cpu().numpy()
            scores = pred_dicts[0]["pred_scores"].detach().cpu().numpy()

            mask = scores >= args.score_thresh
            boxes = boxes[mask]
            scores = scores[mask]

            # Aktuelle ungefilterte Boxen für den nächsten Frame sichern
            current_boxes = boxes.copy()

            # Nur räumlich stabile Detektionen anzeigen
            boxes, scores = temporal_filter(
                boxes=boxes,
                scores=scores,
                previous_boxes=previous_boxes,
                max_lateral_distance=0.35
            )

            previous_boxes = current_boxes

            if len(boxes) > 0:
                dx, dy, dz = boxes[:, 3], boxes[:, 4], boxes[:, 5]
                size_mask = (
                    (dx > 0.05) & (dx < 0.60) &
                    (dy > 0.05) & (dy < 0.60) &
                    (dz > 0.05) & (dz < 0.90)
                )
                boxes = boxes[size_mask]
                scores = scores[size_mask]

            pipeline_time = time.time() - frame_start
            pipeline_fps = 1.0 / max(pipeline_time, 1e-6)
            img = draw_bev(
                display_points,
                boxes,
                scores,
                frame_id,
                model_fps,
                pipeline_fps,
                args
            )

            writer.write(img)
            print(f"Frame {frame_id:04d} | boxes={len(boxes)} | model_fps={model_fps:.2f} | pipeline_fps={pipeline_fps:.2f}")

    writer.release()
    print(f"Video gespeichert: {args.output}")


if __name__ == "__main__":
    main()
