"""OpenEarthMap label definitions."""

CLASS_NAMES = [
    "background",
    "bareland",
    "rangeland",
    "developed",
    "road",
    "tree",
    "water",
    "agriculture",
    "building",
]

# OpenEarthMap's published class colors, with black reserved for background.
CLASS_COLORS = [
    (0, 0, 0),
    (128, 0, 0),
    (0, 255, 36),
    (148, 148, 148),
    (255, 255, 255),
    (34, 97, 38),
    (0, 69, 255),
    (75, 181, 73),
    (222, 31, 7),
]
NUM_CLASSES = len(CLASS_NAMES)
