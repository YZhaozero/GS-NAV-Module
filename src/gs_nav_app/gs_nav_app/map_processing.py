"""Local point-cloud and occupancy-grid map processing utilities.

The desktop UI deliberately keeps these operations local: no HTTP bridge and no
background web service are required.  PCD support covers ASCII, binary, and the
``binary_compressed`` layout produced by PCL. PLY support covers ASCII and
little/big-endian binary vertex data.
"""

from __future__ import annotations

import io
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import yaml


@dataclass
class PointCloudMap:
    points: np.ndarray
    path: Optional[Path] = None
    frame_id: str = "map"
    colors: Optional[np.ndarray] = None
    color_source: str = ""
    splat_scales: Optional[np.ndarray] = None
    splat_rotations: Optional[np.ndarray] = None
    splat_opacities: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        points = np.asarray(self.points, dtype=np.float32)
        if points.ndim != 2 or points.shape[1] < 3:
            raise ValueError("点云数据必须是 N×3 数组")
        self.points = np.ascontiguousarray(points[:, :3])
        if self.colors is not None:
            colors = np.asarray(self.colors, dtype=np.uint8)
            if colors.ndim != 2 or colors.shape != (len(points), 3):
                raise ValueError("点云颜色必须是与点数量一致的 N×3 数组")
            self.colors = np.ascontiguousarray(colors)
            if not self.color_source:
                self.color_source = "rgb"
        elif self.color_source:
            self.color_source = ""
        for name, width in (("splat_scales", 3), ("splat_rotations", 4)):
            value = getattr(self, name)
            if value is not None:
                array = np.asarray(value, dtype=np.float32)
                if array.shape != (len(points), width):
                    raise ValueError(f"{name} 必须是 N×{width} 数组")
                setattr(self, name, np.ascontiguousarray(array))
        if self.splat_opacities is not None:
            opacities = np.asarray(self.splat_opacities, dtype=np.float32).reshape(-1)
            if len(opacities) != len(points):
                raise ValueError("splat_opacities 必须与点数量一致")
            self.splat_opacities = np.ascontiguousarray(opacities)

    @property
    def is_gaussian_splat(self) -> bool:
        return (
            self.splat_scales is not None
            and self.splat_rotations is not None
            and self.splat_opacities is not None
        )


@dataclass
class GridMap:
    occupancy: np.ndarray
    resolution: float
    origin: np.ndarray
    path: Optional[Path] = None
    frame_id: str = "map"

    def __post_init__(self) -> None:
        occupancy = np.asarray(self.occupancy, dtype=np.int8)
        if occupancy.ndim != 2 or not occupancy.size:
            raise ValueError("栅格地图必须是非空二维数组")
        if self.resolution <= 0.0:
            raise ValueError("地图分辨率必须大于 0")
        origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        if len(origin) < 2:
            raise ValueError("地图原点至少需要 x、y")
        if len(origin) == 2:
            origin = np.append(origin, 0.0)
        self.occupancy = np.ascontiguousarray(occupancy)
        self.origin = origin[:3]


_PCD_TYPES = {
    ("F", 4): "<f4",
    ("F", 8): "<f8",
    ("I", 1): "<i1",
    ("I", 2): "<i2",
    ("I", 4): "<i4",
    ("I", 8): "<i8",
    ("U", 1): "<u1",
    ("U", 2): "<u2",
    ("U", 4): "<u4",
    ("U", 8): "<u8",
}

_PLY_TYPES = {
    "char": "i1",
    "int8": "i1",
    "uchar": "u1",
    "uint8": "u1",
    "short": "i2",
    "int16": "i2",
    "ushort": "u2",
    "uint16": "u2",
    "int": "i4",
    "int32": "i4",
    "uint": "u4",
    "uint32": "u4",
    "float": "f4",
    "float32": "f4",
    "double": "f8",
    "float64": "f8",
}


def _read_pcd_header(stream) -> Tuple[Dict[str, list[str]], str]:
    header: Dict[str, list[str]] = {}
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PCD 文件缺少 DATA 头")
        decoded = line.decode("ascii", errors="strict").strip()
        if not decoded or decoded.startswith("#"):
            continue
        parts = decoded.split()
        key = parts[0].upper()
        header[key] = parts[1:]
        if key == "DATA":
            return header, parts[1].lower()


def _pcd_layout(header: Dict[str, list[str]]) -> Tuple[list, int]:
    fields = header.get("FIELDS") or header.get("FIELD")
    if not fields:
        raise ValueError("PCD 文件缺少 FIELDS")
    sizes = [int(value) for value in header.get("SIZE", [])]
    types = [value.upper() for value in header.get("TYPE", [])]
    counts = [int(value) for value in header.get("COUNT", ["1"] * len(fields))]
    if not (len(fields) == len(sizes) == len(types) == len(counts)):
        raise ValueError("PCD FIELDS/SIZE/TYPE/COUNT 长度不一致")
    layout = []
    for name, size, kind, count in zip(fields, sizes, types, counts):
        dtype = _PCD_TYPES.get((kind, size))
        if dtype is None:
            raise ValueError(f"不支持的 PCD 字段类型: {name} {kind}{size}")
        layout.append((name, np.dtype(dtype), count))
    points = int((header.get("POINTS") or header.get("WIDTH") or ["0"])[0])
    return layout, points


def _xyz_columns(layout: Iterable[tuple]) -> Tuple[int, int, int]:
    offsets = {}
    column = 0
    for name, _dtype, count in layout:
        offsets[name.lower()] = column
        column += count
    try:
        return offsets["x"], offsets["y"], offsets["z"]
    except KeyError as exc:
        raise ValueError("PCD 文件必须包含 x、y、z 字段") from exc


def _packed_rgb_to_colors(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values).reshape(-1)
    if values.dtype.kind == "f":
        if values.dtype.itemsize != 4:
            return np.empty((0, 3), dtype=np.uint8)
        packed = np.ascontiguousarray(values.astype("<f4", copy=False)).view("<u4")
    else:
        packed = values.astype("<u4", copy=False)
    return np.ascontiguousarray(np.column_stack([
        (packed >> 16) & 0xFF,
        (packed >> 8) & 0xFF,
        packed & 0xFF,
    ]).astype(np.uint8))


def _colors_from_columns(columns: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    names = {name.lower(): values for name, values in columns.items()}
    packed_name = "rgba" if "rgba" in names else "rgb" if "rgb" in names else None
    if packed_name is not None:
        colors = _packed_rgb_to_colors(names[packed_name])
        return colors if len(colors) else None
    if all(axis in names for axis in ("r", "g", "b")):
        return np.ascontiguousarray(np.column_stack([
            np.asarray(names[axis]).reshape(-1) for axis in ("r", "g", "b")
        ]).astype(np.uint8))
    if "intensity" in names:
        intensity = np.asarray(names["intensity"], dtype=np.float64).reshape(-1)
        finite = np.isfinite(intensity)
        if not np.any(finite):
            return None
        low, high = np.percentile(intensity[finite], [2.0, 98.0])
        if high - low < 1e-9:
            grayscale = np.full(len(intensity), 210, dtype=np.uint8)
        else:
            normalized = np.zeros(len(intensity), dtype=np.float64)
            normalized[finite] = np.clip(
                (intensity[finite] - low) / (high - low), 0.0, 1.0)
            grayscale = np.clip(
                np.rint(35.0 + normalized * 220.0), 0, 255).astype(np.uint8)
            grayscale[~finite] = 35
        return np.ascontiguousarray(np.repeat(grayscale[:, None], 3, axis=1))
    return None


def _pcd_color_source(columns: Dict[str, np.ndarray]) -> str:
    names = {name.lower() for name in columns}
    if "rgb" in names or "rgba" in names or {"r", "g", "b"} <= names:
        return "rgb"
    if "intensity" in names:
        return "intensity"
    return ""


def _lzf_decompress(data: bytes, expected_size: int) -> bytes:
    """Small LZF decoder used by PCL's binary_compressed PCD format."""
    output = bytearray()
    index = 0
    while index < len(data):
        control = data[index]
        index += 1
        if control < 32:
            length = control + 1
            output.extend(data[index:index + length])
            index += length
            continue
        length = control >> 5
        reference = len(output) - ((control & 0x1F) << 8) - 1
        if length == 7:
            if index >= len(data):
                raise ValueError("损坏的 LZF 数据")
            length += data[index]
            index += 1
        if index >= len(data):
            raise ValueError("损坏的 LZF 数据")
        reference -= data[index]
        index += 1
        length += 2
        if reference < 0:
            raise ValueError("损坏的 LZF 回溯偏移")
        for _ in range(length):
            output.append(output[reference])
            reference += 1
    if len(output) != expected_size:
        raise ValueError(
            f"PCD 解压长度错误: {len(output)} != {expected_size}")
    return bytes(output)


def load_pcd(path: str | Path) -> PointCloudMap:
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        header, encoding = _read_pcd_header(stream)
        layout, point_count = _pcd_layout(header)
        payload = stream.read()

    if encoding == "ascii":
        values = np.loadtxt(io.BytesIO(payload), dtype=np.float64, ndmin=2)
        x_col, y_col, z_col = _xyz_columns(layout)
        points = values[:, [x_col, y_col, z_col]]
        field_arrays = {}
        column = 0
        for name, dtype, count in layout:
            raw_values = values[:, column:column + count]
            field_arrays[name.lower()] = raw_values.astype(dtype, copy=False)
            column += count
        colors = _colors_from_columns(field_arrays)
        color_source = _pcd_color_source(field_arrays)
    elif encoding == "binary":
        names = []
        for name, dtype, count in layout:
            names.append((name, dtype, (count,)) if count > 1 else (name, dtype))
        records = np.frombuffer(payload, dtype=np.dtype(names), count=point_count)
        points = np.column_stack([
            records[next(name for name in records.dtype.names if name.lower() == axis)]
            for axis in ("x", "y", "z")
        ])
        field_arrays = {
            name.lower(): records[name] for name in records.dtype.names
        }
        colors = _colors_from_columns(field_arrays)
        color_source = _pcd_color_source(field_arrays)
    elif encoding == "binary_compressed":
        if len(payload) < 8:
            raise ValueError("PCD 压缩数据头不完整")
        compressed_size, raw_size = struct.unpack_from("<II", payload, 0)
        raw = _lzf_decompress(payload[8:8 + compressed_size], raw_size)
        field_arrays = {}
        offset = 0
        for name, dtype, count in layout:
            byte_count = point_count * dtype.itemsize * count
            block = np.frombuffer(raw[offset:offset + byte_count], dtype=dtype)
            field_arrays[name.lower()] = block.reshape(point_count, count)
            offset += byte_count
        try:
            points = np.column_stack([
                field_arrays[axis][:, 0] for axis in ("x", "y", "z")])
        except KeyError as exc:
            raise ValueError("PCD 文件必须包含 x、y、z 字段") from exc
        colors = _colors_from_columns(field_arrays)
        color_source = _pcd_color_source(field_arrays)
    else:
        raise ValueError(f"不支持的 PCD DATA 类型: {encoding}")

    points = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if point_count and not len(points):
        raise ValueError("PCD 中没有有效的 XYZ 点")
    return PointCloudMap(
        points=points, path=path, colors=colors, color_source=color_source)


def _read_ply_header(stream):
    first = stream.readline().decode("ascii", errors="strict").strip().lower()
    if first != "ply":
        raise ValueError("PLY 文件缺少 ply 文件头")
    encoding = None
    vertex_count = None
    vertex_properties = []
    current_element = None
    while True:
        line = stream.readline()
        if not line:
            raise ValueError("PLY 文件缺少 end_header")
        parts = line.decode("ascii", errors="strict").strip().split()
        if not parts or parts[0].lower() in ("comment", "obj_info"):
            continue
        key = parts[0].lower()
        if key == "format":
            if len(parts) < 3 or parts[2] != "1.0":
                raise ValueError("只支持 PLY 1.0 格式")
            encoding = parts[1].lower()
        elif key == "element":
            if len(parts) != 3:
                raise ValueError("PLY element 文件头格式错误")
            current_element = parts[1].lower()
            if current_element == "vertex":
                vertex_count = int(parts[2])
        elif key == "property" and current_element == "vertex":
            if len(parts) >= 2 and parts[1].lower() == "list":
                raise ValueError("暂不支持 vertex 中的 PLY list 属性")
            if len(parts) != 3:
                raise ValueError("PLY vertex property 文件头格式错误")
            kind, name = parts[1].lower(), parts[2].lower()
            if kind not in _PLY_TYPES:
                raise ValueError(f"不支持的 PLY 属性类型: {kind}")
            vertex_properties.append((name, kind))
        elif key == "end_header":
            break
    if encoding not in ("ascii", "binary_little_endian", "binary_big_endian"):
        raise ValueError(f"不支持的 PLY DATA 类型: {encoding}")
    if vertex_count is None or vertex_count <= 0:
        raise ValueError("PLY 文件没有有效的 vertex 元素")
    if not vertex_properties:
        raise ValueError("PLY 文件没有 vertex 属性")
    return encoding, vertex_count, vertex_properties


def _ply_colors(columns: Dict[str, np.ndarray]) -> Optional[np.ndarray]:
    aliases = (
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    )
    names = next((group for group in aliases if all(key in columns for key in group)), None)
    if names is None:
        if all(f"f_dc_{index}" in columns for index in range(3)):
            # Degree-zero real spherical harmonic used by 3D Gaussian Splatting.
            sh_c0 = 0.28209479177387814
            colors = np.column_stack([
                columns[f"f_dc_{index}"] for index in range(3)
            ])
            colors = (0.5 + sh_c0 * colors) * 255.0
            return np.ascontiguousarray(
                np.clip(np.rint(colors), 0, 255).astype(np.uint8))
        return None
    colors = np.column_stack([columns[name] for name in names])
    if colors.dtype.kind == "f" and colors.size and float(np.nanmax(colors)) <= 1.0:
        colors = colors * 255.0
    return np.ascontiguousarray(np.clip(np.rint(colors), 0, 255).astype(np.uint8))


def _ply_color_source(columns: Dict[str, np.ndarray]) -> str:
    if all(f"f_dc_{index}" in columns for index in range(3)):
        return "gaussian"
    aliases = (
        ("red", "green", "blue"),
        ("r", "g", "b"),
        ("diffuse_red", "diffuse_green", "diffuse_blue"),
    )
    return "rgb" if any(all(key in columns for key in group) for group in aliases) else ""


def _ply_gaussian_attributes(columns: Dict[str, np.ndarray]):
    required = [
        "opacity", "scale_0", "scale_1", "scale_2",
        "rot_0", "rot_1", "rot_2", "rot_3",
    ]
    if not all(name in columns for name in required):
        return None, None, None
    log_scales = np.column_stack([
        columns[f"scale_{index}"] for index in range(3)
    ]).astype(np.float32)
    scales = np.exp(np.clip(log_scales, -20.0, 8.0)).astype(np.float32)
    rotations = np.column_stack([
        columns[f"rot_{index}"] for index in range(4)
    ]).astype(np.float32)
    norms = np.linalg.norm(rotations, axis=1, keepdims=True)
    rotations /= np.maximum(norms, 1e-8)
    raw_opacity = np.asarray(columns["opacity"], dtype=np.float32)
    opacities = 1.0 / (1.0 + np.exp(-np.clip(raw_opacity, -30.0, 30.0)))
    return (
        np.ascontiguousarray(scales),
        np.ascontiguousarray(rotations),
        np.ascontiguousarray(opacities),
    )


def load_ply(path: str | Path) -> PointCloudMap:
    """Load scalar XYZ vertices and optional RGB colors from a PLY file."""
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        encoding, vertex_count, properties = _read_ply_header(stream)
        if encoding == "ascii":
            values = np.loadtxt(
                stream, dtype=np.float64, ndmin=2, max_rows=vertex_count)
            if values.shape != (vertex_count, len(properties)):
                raise ValueError(
                    f"PLY vertex 数据长度错误: {values.shape[0]} != {vertex_count}")
            columns = {
                name: values[:, index] for index, (name, _kind) in enumerate(properties)
            }
        else:
            endian = "<" if encoding == "binary_little_endian" else ">"
            dtype = np.dtype([
                (name, endian + _PLY_TYPES[kind]) for name, kind in properties
            ])
            records = np.fromfile(stream, dtype=dtype, count=vertex_count)
            if len(records) != vertex_count:
                raise ValueError(
                    f"PLY vertex 数据长度错误: {len(records)} != {vertex_count}")
            columns = {name: records[name] for name, _kind in properties}
    try:
        points = np.column_stack([columns[axis] for axis in ("x", "y", "z")])
    except KeyError as exc:
        raise ValueError("PLY 文件必须包含 x、y、z 属性") from exc
    colors = _ply_colors(columns)
    splat_scales, splat_rotations, splat_opacities = _ply_gaussian_attributes(columns)
    points = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(points), axis=1)
    points = points[valid]
    if colors is not None:
        colors = colors[valid]
    if splat_scales is not None:
        splat_scales = splat_scales[valid]
        splat_rotations = splat_rotations[valid]
        splat_opacities = splat_opacities[valid]
    if not len(points):
        raise ValueError("PLY 中没有有效的 XYZ 点")
    return PointCloudMap(
        points=points,
        path=path,
        colors=colors,
        color_source=_ply_color_source(columns),
        splat_scales=splat_scales,
        splat_rotations=splat_rotations,
        splat_opacities=splat_opacities,
    )


def load_pointcloud(path: str | Path) -> PointCloudMap:
    """Load a supported point-cloud file based on its extension."""
    suffix = Path(path).suffix.lower()
    if suffix == ".pcd":
        return load_pcd(path)
    if suffix == ".ply":
        return load_ply(path)
    raise ValueError(f"不支持的点云格式: {suffix or '无扩展名'}（支持 .pcd/.ply）")


def save_pcd(
    path: str | Path,
    points: np.ndarray,
    colors: Optional[np.ndarray] = None,
) -> Path:
    path = Path(path).expanduser()
    if path.suffix.lower() != ".pcd":
        path = path.with_suffix(".pcd")
    path.parent.mkdir(parents=True, exist_ok=True)
    xyz = np.asarray(points, dtype="<f4")[:, :3]
    if colors is None:
        fields = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1"
        payload = np.ascontiguousarray(xyz).tobytes()
    else:
        rgb = np.asarray(colors, dtype=np.uint8)
        if rgb.shape != (len(xyz), 3):
            raise ValueError("点云颜色必须是与点数量一致的 N×3 数组")
        packed = (
            np.uint32(0xFF000000)
            | (rgb[:, 0].astype(np.uint32) << 16)
            | (rgb[:, 1].astype(np.uint32) << 8)
            | rgb[:, 2].astype(np.uint32)
        )
        records = np.empty(
            len(xyz), dtype=[("x", "<f4"), ("y", "<f4"),
                             ("z", "<f4"), ("rgb", "<u4")])
        records["x"], records["y"], records["z"] = xyz.T
        records["rgb"] = packed
        fields = "FIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F U\nCOUNT 1 1 1 1"
        payload = records.tobytes()
    header = (
        "# .PCD v0.7 - Point Cloud Data file format\n"
        f"VERSION 0.7\n{fields}\nWIDTH {len(xyz)}\nHEIGHT 1\n"
        f"VIEWPOINT 0 0 0 1 0 0 0\nPOINTS {len(xyz)}\nDATA binary\n"
    ).encode("ascii")
    with path.open("wb") as stream:
        stream.write(header)
        stream.write(payload)
    return path.resolve()


def pointcloud_to_grid(
    points: np.ndarray,
    resolution: float = 0.05,
    z_min: float = -0.2,
    z_max: float = 1.5,
    padding: float = 0.5,
    inflation_radius: float = 0.06,
    max_cells: int = 25_000_000,
) -> GridMap:
    if resolution <= 0.0:
        raise ValueError("栅格分辨率必须大于 0")
    xyz = np.asarray(points, dtype=np.float32)
    valid = np.all(np.isfinite(xyz[:, :3]), axis=1)
    valid &= (xyz[:, 2] >= z_min) & (xyz[:, 2] <= z_max)
    selected = xyz[valid]
    if not len(selected):
        raise ValueError("指定高度范围内没有点，无法转换栅格地图")
    lower = np.floor((selected[:, :2].min(axis=0) - padding) / resolution) * resolution
    upper = np.ceil((selected[:, :2].max(axis=0) + padding) / resolution) * resolution
    width, height = np.ceil((upper - lower) / resolution).astype(int) + 1
    if width <= 0 or height <= 0 or int(width) * int(height) > max_cells:
        raise ValueError(
            f"转换结果过大: {width}×{height}，请增大分辨率或缩小点云范围")
    occupancy = np.zeros((int(height), int(width)), dtype=np.int8)
    indices = np.floor((selected[:, :2] - lower) / resolution).astype(int)
    indices[:, 0] = np.clip(indices[:, 0], 0, width - 1)
    indices[:, 1] = np.clip(indices[:, 1], 0, height - 1)
    occupancy[indices[:, 1], indices[:, 0]] = 100

    radius_cells = max(0, int(math.ceil(inflation_radius / resolution)))
    if radius_cells:
        source = occupancy == 100
        inflated = source.copy()
        for dy in range(-radius_cells, radius_cells + 1):
            for dx in range(-radius_cells, radius_cells + 1):
                if dx * dx + dy * dy > radius_cells * radius_cells:
                    continue
                source_y0, source_y1 = max(0, -dy), min(height, height - dy)
                source_x0, source_x1 = max(0, -dx), min(width, width - dx)
                target_y0, target_y1 = source_y0 + dy, source_y1 + dy
                target_x0, target_x1 = source_x0 + dx, source_x1 + dx
                inflated[target_y0:target_y1, target_x0:target_x1] |= source[
                    source_y0:source_y1, source_x0:source_x1]
        occupancy[inflated] = 100
    return GridMap(occupancy, float(resolution), np.array([*lower, 0.0]))


def _pgm_tokens(data: bytes, count: int = 4) -> Tuple[list[bytes], int]:
    tokens = []
    index = 0
    while len(tokens) < count:
        while index < len(data) and chr(data[index]).isspace():
            index += 1
        if index < len(data) and data[index] == ord("#"):
            while index < len(data) and data[index] not in (10, 13):
                index += 1
            continue
        start = index
        while index < len(data) and not chr(data[index]).isspace():
            index += 1
        if start == index:
            raise ValueError("PGM 文件头不完整")
        tokens.append(data[start:index])
    return tokens, index


def read_pgm(path: str | Path) -> np.ndarray:
    data = Path(path).read_bytes()
    tokens, offset = _pgm_tokens(data)
    magic, width_raw, height_raw, max_value_raw = tokens
    width, height, max_value = int(width_raw), int(height_raw), int(max_value_raw)
    if max_value <= 0 or max_value > 65535:
        raise ValueError("不支持的 PGM 灰度范围")
    if magic == b"P5":
        if offset >= len(data) or not chr(data[offset]).isspace():
            raise ValueError("PGM 文件头后缺少分隔符")
        if data[offset:offset + 2] == b"\r\n":
            offset += 2
        else:
            offset += 1
        dtype = np.uint8 if max_value < 256 else np.dtype(">u2")
        image = np.frombuffer(data, dtype=dtype, count=width * height, offset=offset)
    elif magic == b"P2":
        image = np.fromstring(data[offset:].decode("ascii"), sep=" ", dtype=np.uint16)
    else:
        raise ValueError("只支持 P2/P5 PGM 文件")
    if image.size != width * height:
        raise ValueError("PGM 像素数据长度错误")
    if max_value != 255:
        image = np.rint(image.astype(np.float64) * 255.0 / max_value)
    return image.reshape(height, width).astype(np.uint8)


def load_grid_map(path: str | Path, default_resolution: float = 0.05) -> GridMap:
    path = Path(path).expanduser().resolve()
    config = {}
    if path.suffix.lower() in (".yaml", ".yml"):
        config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        image_path = Path(str(config.get("image", ""))).expanduser()
        if not image_path.is_absolute():
            image_path = path.parent / image_path
    else:
        image_path = path
    pixels = read_pgm(image_path)
    negate = bool(int(config.get("negate", 0)))
    occupied_thresh = float(config.get("occupied_thresh", 0.65))
    free_thresh = float(config.get("free_thresh", 0.196))
    probability = pixels.astype(np.float32) / 255.0
    if not negate:
        probability = 1.0 - probability
    occupancy = np.full(pixels.shape, -1, dtype=np.int8)
    occupancy[probability > occupied_thresh] = 100
    occupancy[probability < free_thresh] = 0
    # PGM rows start at the top; OccupancyGrid rows start at the map origin.
    occupancy = np.flipud(occupancy).copy()
    origin = np.asarray(config.get("origin", [0.0, 0.0, 0.0]), dtype=float)
    resolution = float(config.get("resolution", default_resolution))
    return GridMap(occupancy, resolution, origin, path=path)


def save_grid_map(path: str | Path, grid: GridMap) -> Tuple[Path, Path]:
    yaml_path = Path(path).expanduser()
    if yaml_path.suffix.lower() not in (".yaml", ".yml"):
        yaml_path = yaml_path.with_suffix(".yaml")
    pgm_path = yaml_path.with_suffix(".pgm")
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    display = np.flipud(grid.occupancy)
    pixels = np.full(display.shape, 205, dtype=np.uint8)
    pixels[display == 0] = 254
    pixels[display >= 50] = 0
    height, width = pixels.shape
    with pgm_path.open("wb") as stream:
        stream.write(f"P5\n# Created by gs_nav_app\n{width} {height}\n255\n".encode("ascii"))
        stream.write(pixels.tobytes())
    payload = {
        "image": pgm_path.name,
        "mode": "trinary",
        "resolution": float(grid.resolution),
        "origin": [float(value) for value in grid.origin],
        "negate": 0,
        "occupied_thresh": 0.65,
        "free_thresh": 0.196,
    }
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return yaml_path.resolve(), pgm_path.resolve()


def pointcloud2_to_xyz(message, max_points: int = 400_000) -> np.ndarray:
    """Extract XYZ from sensor_msgs/PointCloud2 without sensor_msgs_py."""
    fields = {field.name: field for field in message.fields}
    if not all(axis in fields for axis in ("x", "y", "z")):
        raise ValueError("PointCloud2 缺少 x/y/z 字段")
    datatype_map = {
        1: "i1", 2: "u1", 3: "<i2", 4: "<u2",
        5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8",
    }
    names, formats, offsets = [], [], []
    for axis in ("x", "y", "z"):
        field = fields[axis]
        if field.datatype not in datatype_map:
            raise ValueError(f"不支持的 PointCloud2 {axis} 数据类型")
        names.append(axis)
        formats.append(datatype_map[field.datatype])
        offsets.append(int(field.offset))
    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": int(message.point_step),
    })
    count = int(message.width) * int(message.height)
    records = np.frombuffer(message.data, dtype=dtype, count=count)
    if count > max_points:
        records = records[::max(1, int(math.ceil(count / max_points)))]
    points = np.column_stack([records[axis] for axis in ("x", "y", "z")])
    points = np.asarray(points, dtype=np.float32)
    return points[np.all(np.isfinite(points), axis=1)]
