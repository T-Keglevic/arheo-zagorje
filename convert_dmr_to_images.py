"""
DMR TIF to High-Resolution Image Converter - Version 3
ARCHAEOLOGICAL DISCOVERY TOOL

Converts Digital Terrain Model (DMR) .tif files to FULL RESOLUTION images
optimized for detecting archaeological features:
- Burial mounds, tumuli
- Ancient roads and paths
- Field boundaries and lynchets  
- Building foundations
- Ditches and defensive earthworks
- Subtle terrain anomalies

Features:
- Full native resolution output (no downsampling)
- Multi-directional hillshade with vertical exaggeration
- Local Relief Model for subtle feature detection
- Interactive HTML viewer with:
  - Pan/zoom on full-res images
  - Geographic navigation (arrow keys = N/S/E/W)
  - Real-time brightness/contrast adjustment
  - Coordinate tracking

Usage:
    python convert_dmr_to_images.py --input /path/to/tifs --output /path/to/output

Requirements:
    pip install rasterio numpy pillow matplotlib tqdm scipy

Author: Claude
Version: 3.0
"""

import argparse
import os
import sys
import json
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')


def install_dependencies():
    """Install required packages if not present."""
    required = ['rasterio', 'numpy', 'pillow', 'matplotlib', 'tqdm', 'scipy']
    import subprocess
    for package in required:
        try:
            __import__(package if package != 'pillow' else 'PIL')
        except ImportError:
            print(f"Installing {package}...")
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package, '-q'])


try:
    import rasterio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import cm
    from tqdm import tqdm
    from scipy.ndimage import uniform_filter, gaussian_filter, sobel
except ImportError:
    install_dependencies()
    import rasterio
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from matplotlib import cm
    from tqdm import tqdm
    from scipy.ndimage import uniform_filter, gaussian_filter, sobel


# =============================================================================
# ARCHAEOLOGICAL VISUALIZATION ALGORITHMS
# =============================================================================

def calculate_hillshade(elevation, azimuth=315, altitude=45, z_factor=1.0, cell_size=1.0):
    """
    Calculate hillshade with configurable vertical exaggeration.
    Uses Sobel filters for more accurate gradient estimation.
    
    Args:
        elevation: 2D array of elevation values
        azimuth: Light source direction (degrees from north)
        altitude: Light source angle above horizon (degrees)
        z_factor: Vertical exaggeration multiplier
        cell_size: Size of each cell (for proper gradient scaling)
    """
    azimuth_rad = np.radians(360 - azimuth + 90)
    altitude_rad = np.radians(altitude)
    
    # Use Sobel filters for better gradient estimation (less noise than np.gradient)
    # Sobel gives weighted average of surrounding cells
    dx = sobel(elevation, axis=1, mode='reflect') / (8.0 * cell_size) * z_factor
    dy = sobel(elevation, axis=0, mode='reflect') / (8.0 * cell_size) * z_factor
    
    slope = np.arctan(np.sqrt(dx**2 + dy**2))
    aspect = np.arctan2(-dy, dx)
    
    hillshade = (np.cos(altitude_rad) * np.cos(slope) + 
                 np.sin(altitude_rad) * np.sin(slope) * np.cos(azimuth_rad - aspect))
    
    return np.clip((hillshade + 1) / 2, 0, 1)


def calculate_multidirectional_hillshade(elevation, z_factor=3.0, cell_size=1.0):
    """
    Multi-directional hillshade - essential for archaeology.
    
    Uses 16 light directions to reveal features at any orientation.
    Low sun angles (15°, 25°, 35°) emphasize subtle relief.
    """
    # More directions and lower angles for better feature detection
    azimuths = [0, 22.5, 45, 67.5, 90, 112.5, 135, 157.5, 180, 202.5, 225, 247.5, 270, 292.5, 315, 337.5]
    altitudes = [15, 25, 35, 45]  # Very low to medium angles
    
    # Weight lower angles more heavily - they show more subtle detail
    altitude_weights = {15: 1.5, 25: 1.3, 35: 1.0, 45: 0.7}
    
    hillshades = []
    weights = []
    
    for alt in altitudes:
        weight = altitude_weights[alt]
        for az in azimuths:
            hs = calculate_hillshade(elevation, azimuth=az, altitude=alt, 
                                    z_factor=z_factor, cell_size=cell_size)
            hillshades.append(hs)
            weights.append(weight)
    
    # Weighted combination
    weights = np.array(weights)
    combined = np.average(hillshades, axis=0, weights=weights)
    
    return combined


def calculate_slope(elevation, cell_size=1.0):
    """Calculate slope in degrees using Sobel for accuracy."""
    dx = sobel(elevation, axis=1, mode='reflect') / (8.0 * cell_size)
    dy = sobel(elevation, axis=0, mode='reflect') / (8.0 * cell_size)
    return np.degrees(np.arctan(np.sqrt(dx**2 + dy**2)))


def calculate_local_relief_model(elevation, kernel_size=25):
    """
    Local Relief Model (LRM) - highlights subtle local variations.
    
    Subtracts a smoothed version from the original, revealing
    small-scale features that get lost in regional elevation trends.
    """
    # Use float64 for precision
    elev = elevation.astype(np.float64)
    local_mean = uniform_filter(elev, size=kernel_size, mode='reflect')
    return elev - local_mean


def calculate_multi_scale_lrm(elevation):
    """
    Multi-scale Local Relief Model.
    
    Different kernel sizes reveal features at different scales:
    - Small kernel (11): fine details, small pits, postholes
    - Medium kernel (25): typical archaeological features
    - Large kernel (51): larger structures, field boundaries
    """
    lrm_small = calculate_local_relief_model(elevation, kernel_size=11)
    lrm_medium = calculate_local_relief_model(elevation, kernel_size=25)
    lrm_large = calculate_local_relief_model(elevation, kernel_size=51)
    
    # Combine with weights favoring medium scale
    combined = 0.25 * lrm_small + 0.50 * lrm_medium + 0.25 * lrm_large
    return combined


def calculate_skyview_factor_approx(elevation, radius=15):
    """
    Improved Sky-View Factor approximation using multiple scales.
    """
    # Multi-scale difference of gaussians
    blur1 = gaussian_filter(elevation.astype(np.float64), sigma=radius/4)
    blur2 = gaussian_filter(elevation.astype(np.float64), sigma=radius/2)
    blur3 = gaussian_filter(elevation.astype(np.float64), sigma=radius)
    blur4 = gaussian_filter(elevation.astype(np.float64), sigma=radius*2)
    
    # Combine differences at multiple scales
    dog1 = blur1 - blur2
    dog2 = blur2 - blur3
    dog3 = blur3 - blur4
    
    svf = 0.5 * dog1 + 0.35 * dog2 + 0.15 * dog3
    return svf


def adaptive_histogram_equalization(arr, clip_limit=0.03):
    """
    Contrast Limited Adaptive Histogram Equalization.
    Enhances local contrast without over-amplifying noise.
    
    Operates on float array, returns float array.
    """
    # Normalize to 0-1
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    if arr_max <= arr_min:
        return arr
    
    normalized = (arr - arr_min) / (arr_max - arr_min)
    
    # Process in overlapping tiles
    tile_size = 64
    h, w = normalized.shape
    result = np.zeros_like(normalized)
    weight = np.zeros_like(normalized)
    
    for y in range(0, h, tile_size // 2):
        for x in range(0, w, tile_size // 2):
            y2 = min(y + tile_size, h)
            x2 = min(x + tile_size, w)
            
            tile = normalized[y:y2, x:x2]
            
            # Simple histogram equalization per tile
            flat = tile.flatten()
            flat_sorted = np.sort(flat)
            
            # Create mapping
            ranks = np.searchsorted(flat_sorted, tile)
            equalized = ranks.astype(np.float64) / max(1, ranks.max())
            
            # Blend with original (clip_limit controls strength)
            blended = (1 - clip_limit) * tile + clip_limit * equalized
            
            # Accumulate with simple weighting
            result[y:y2, x:x2] += blended
            weight[y:y2, x:x2] += 1
    
    result /= np.maximum(weight, 1)
    return result


def normalize_array(arr, percentile_low=1, percentile_high=99):
    """Normalize array to 0-1 with percentile clipping for robustness."""
    valid = arr[np.isfinite(arr)]
    if len(valid) == 0:
        return np.zeros_like(arr)
    
    vmin = np.percentile(valid, percentile_low)
    vmax = np.percentile(valid, percentile_high)
    
    if vmax <= vmin:
        return np.zeros_like(arr)
    
    normalized = (arr - vmin) / (vmax - vmin)
    return np.clip(normalized, 0, 1)


def create_archaeological_visualization(elevation, z_factor=3.0, cell_size=1.0, high_quality=True):
    """
    Create visualization optimized for archaeological feature detection.
    
    HIGH QUALITY MODE (default):
    - 16 hillshade directions × 4 altitude angles
    - Multi-scale Local Relief Model
    - Sky-View Factor approximation
    - Adaptive contrast enhancement
    - 16-bit internal processing
    
    Output is high-contrast grayscale optimized for human pattern recognition.
    Returns uint16 array for maximum quality, or uint8 if high_quality=False.
    """
    # Handle invalid data
    valid_mask = np.isfinite(elevation)
    if not np.any(valid_mask):
        return np.zeros((*elevation.shape, 3), dtype=np.uint8)
    
    # Work in float64 for maximum precision
    elev = elevation.astype(np.float64)
    elev_filled = elev.copy()
    elev_filled[~valid_mask] = np.nanmedian(elev)  # median is more robust than mean
    
    # 1. Multi-directional hillshade (primary layer) - 16 directions × 4 angles
    hillshade = calculate_multidirectional_hillshade(elev_filled, z_factor=z_factor, cell_size=cell_size)
    
    # 2. Multi-scale Local Relief Model
    ms_lrm = calculate_multi_scale_lrm(elev_filled)
    ms_lrm_norm = normalize_array(ms_lrm, percentile_low=0.5, percentile_high=99.5)
    
    # 3. Sky-View Factor approximation
    svf = calculate_skyview_factor_approx(elev_filled, radius=15)
    svf_norm = normalize_array(svf, percentile_low=1, percentile_high=99)
    
    # 4. Slope for edge definition
    slope = calculate_slope(elev_filled, cell_size=cell_size)
    slope_norm = normalize_array(slope, percentile_low=0, percentile_high=98)
    
    # Combine layers - weights tuned for archaeological feature detection
    combined = (
        0.45 * hillshade +       # Main terrain shape and shadows
        0.30 * ms_lrm_norm +     # Subtle local anomalies at multiple scales
        0.15 * svf_norm +        # Depressions and enclosures
        0.10 * (1 - slope_norm)  # Soft edge enhancement
    )
    
    # Apply adaptive contrast enhancement
    combined = adaptive_histogram_equalization(combined, clip_limit=0.02)
    
    # Final normalization - use tight percentiles to maximize dynamic range
    combined = normalize_array(combined, percentile_low=0.1, percentile_high=99.9)
    
    # Apply subtle gamma for better mid-tone visibility
    gamma = 0.95
    combined = np.power(np.clip(combined, 0, 1), gamma)
    
    # Mask invalid areas
    combined[~valid_mask] = 0.5
    
    if high_quality:
        # 16-bit output for maximum quality
        gray16 = (combined * 65535).astype(np.uint16)
        # Return as single channel - will be converted to RGB later if needed
        return gray16
    else:
        # 8-bit output for web viewer
        gray8 = (combined * 255).astype(np.uint8)
        rgb = np.stack([gray8, gray8, gray8], axis=-1)
        return rgb


def create_standard_visualization(elevation, colormap='terrain', z_factor=2.0, cell_size=1.0):
    """Standard colored terrain visualization with hillshade."""
    valid_mask = np.isfinite(elevation)
    elev_filled = elevation.astype(np.float64).copy()
    elev_filled[~valid_mask] = np.nanmedian(elevation)
    
    # Normalize elevation for colormap
    elev_norm = normalize_array(elev_filled)
    
    # Apply colormap
    cmap = cm.get_cmap(colormap)
    colored = cmap(elev_norm)[:, :, :3]
    
    # Blend with hillshade
    hillshade = calculate_multidirectional_hillshade(elev_filled, z_factor=z_factor, cell_size=cell_size)
    hillshade_3d = np.stack([hillshade] * 3, axis=-1)
    
    # Overlay blend
    result = np.where(
        hillshade_3d < 0.5,
        2 * colored * hillshade_3d,
        1 - 2 * (1 - colored) * (1 - hillshade_3d)
    )
    
    result = np.clip(result, 0, 1)
    result[~valid_mask] = 0.5
    
    return (result * 255).astype(np.uint8)


# =============================================================================
# FILE PROCESSING
# =============================================================================

def get_tif_metadata(tif_path):
    """Extract metadata from TIF file."""
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            return {
                'path': str(tif_path),
                'name': tif_path.stem,
                'bounds': {
                    'left': bounds.left,
                    'bottom': bounds.bottom,
                    'right': bounds.right,
                    'top': bounds.top
                },
                'center': {
                    'x': (bounds.left + bounds.right) / 2,
                    'y': (bounds.bottom + bounds.top) / 2
                },
                'width': src.width,
                'height': src.height,
                'crs': str(src.crs) if src.crs else None,
                'pixel_size': {
                    'x': abs(src.transform[0]),
                    'y': abs(src.transform[4])
                }
            }
    except Exception as e:
        return {'path': str(tif_path), 'name': tif_path.stem, 'error': str(e)}


def convert_single_tif(args):
    """
    Convert a single TIF to full-resolution image.
    
    NO DOWNSAMPLING - preserves every pixel of the source data.
    
    For archaeological mode:
    - Produces 16-bit grayscale PNG for maximum quality
    - Web viewer will still work (browsers handle 16-bit PNG)
    """
    tif_path, output_path, output_format, mode, colormap, z_factor, quality = args
    
    try:
        with rasterio.open(tif_path) as src:
            # Read at FULL RESOLUTION
            elevation = src.read(1).astype(np.float32)
            
            # Get cell size for proper gradient calculation
            cell_size = abs(src.transform[0])  # Pixel size in map units
            
            # Handle nodata
            if src.nodata is not None:
                elevation[elevation == src.nodata] = np.nan
            
            img_width = src.width
            img_height = src.height
        
        # Create visualization at full resolution
        if mode == 'archaeological':
            # Get 16-bit result for maximum quality
            result = create_archaeological_visualization(
                elevation, z_factor=z_factor, cell_size=cell_size, high_quality=True
            )
            
            if output_format.lower() == 'png':
                # Save as 16-bit grayscale PNG for maximum quality
                img = Image.fromarray(result, mode='I;16')
                img.save(output_path, 'PNG', compress_level=6)
            else:
                # JPEG only supports 8-bit, convert
                result_8bit = (result / 256).astype(np.uint8)
                rgb = np.stack([result_8bit, result_8bit, result_8bit], axis=-1)
                img = Image.fromarray(rgb)
                img.save(output_path, 'JPEG', quality=quality, subsampling=0)
        else:
            # Standard mode - 8-bit color
            rgb = create_standard_visualization(
                elevation, colormap=colormap, z_factor=z_factor, cell_size=cell_size
            )
            img = Image.fromarray(rgb)
            
            if output_format.lower() == 'png':
                img.save(output_path, 'PNG', compress_level=6)
            else:
                img.save(output_path, 'JPEG', quality=quality, subsampling=0)
        
        return (True, f"OK: {tif_path.name} ({img_width}x{img_height})")
    
    except Exception as e:
        import traceback
        return (False, f"ERROR {tif_path.name}: {str(e)}\n{traceback.format_exc()}")


def find_tif_files(input_dir):
    """Find all TIF files in directory."""
    input_path = Path(input_dir)
    patterns = ['*.tif', '*.tiff', '*.TIF', '*.TIFF']
    
    tif_files = []
    for pattern in patterns:
        tif_files.extend(input_path.glob(pattern))
    
    return sorted(set(tif_files))


# =============================================================================
# OVERVIEW MAP GENERATION
# =============================================================================

def create_overview_map(tile_metadata, output_path, max_size=4000):
    """
    Create overview map for navigation.
    
    This is intentionally lower resolution - it's just for clicking to navigate.
    The actual tile images are full resolution.
    """
    print("\nGenerating overview map...")
    
    valid_tiles = [t for t in tile_metadata if 'error' not in t]
    if not valid_tiles:
        return None
    
    # Calculate bounds
    min_x = min(t['bounds']['left'] for t in valid_tiles)
    max_x = max(t['bounds']['right'] for t in valid_tiles)
    min_y = min(t['bounds']['bottom'] for t in valid_tiles)
    max_y = max(t['bounds']['top'] for t in valid_tiles)
    
    total_width = max_x - min_x
    total_height = max_y - min_y
    
    # Calculate overview dimensions
    if total_width > total_height:
        overview_width = max_size
        overview_height = int(max_size * total_height / total_width)
    else:
        overview_height = max_size
        overview_width = int(max_size * total_width / total_height)
    
    overview = np.zeros((overview_height, overview_width, 3), dtype=np.uint8)
    
    for tile in tqdm(valid_tiles, desc="Building overview"):
        try:
            with rasterio.open(tile['path']) as src:
                # Read at reduced resolution for overview only
                target_pixels = 150  # ~150px per tile in overview
                scale = max(1, min(src.width, src.height) // target_pixels)
                out_shape = (max(1, src.height // scale), max(1, src.width // scale))
                
                elevation = src.read(1, out_shape=out_shape).astype(np.float32)
                if src.nodata is not None:
                    elevation[elevation == src.nodata] = np.nan
                
                # Use 8-bit mode for overview (faster, smaller)
                tile_result = create_archaeological_visualization(elevation, z_factor=2.0, high_quality=False)
                
                # Position in overview
                x1 = int((tile['bounds']['left'] - min_x) / total_width * overview_width)
                x2 = int((tile['bounds']['right'] - min_x) / total_width * overview_width)
                y1 = int((max_y - tile['bounds']['top']) / total_height * overview_height)
                y2 = int((max_y - tile['bounds']['bottom']) / total_height * overview_height)
                
                w, h = max(1, x2 - x1), max(1, y2 - y1)
                
                tile_img = Image.fromarray(tile_result).resize((w, h), Image.Resampling.LANCZOS)
                overview[y1:y1+h, x1:x1+w] = np.array(tile_img)
                
        except Exception as e:
            print(f"Warning: {tile['name']}: {e}")
    
    # Save overview
    Image.fromarray(overview).save(output_path / 'overview_map.png', 'PNG')
    print(f"Overview map saved: {output_path / 'overview_map.png'}")
    
    return {
        'bounds': {'min_x': min_x, 'max_x': max_x, 'min_y': min_y, 'max_y': max_y},
        'size': {'width': overview_width, 'height': overview_height}
    }


# =============================================================================
# GEOGRAPHIC GRID BUILDING
# =============================================================================

def build_geographic_grid(tile_metadata):
    """
    Build a geographic grid mapping for spatial navigation.
    
    Determines which tiles are neighbors (N/S/E/W) based on coordinates.
    """
    valid_tiles = [t for t in tile_metadata if 'error' not in t]
    
    # Get unique X and Y positions with tolerance
    def round_coord(val, precision=1):
        return round(val, precision)
    
    # Group tiles by their grid position
    x_coords = sorted(set(round_coord(t['bounds']['left']) for t in valid_tiles))
    y_coords = sorted(set(round_coord(t['bounds']['top']) for t in valid_tiles), reverse=True)
    
    # Create grid mapping
    grid = {}
    tile_to_grid = {}
    
    for tile in valid_tiles:
        tx = round_coord(tile['bounds']['left'])
        ty = round_coord(tile['bounds']['top'])
        
        col = x_coords.index(tx) if tx in x_coords else -1
        row = y_coords.index(ty) if ty in y_coords else -1
        
        if col >= 0 and row >= 0:
            grid[(row, col)] = tile['name']
            tile_to_grid[tile['name']] = (row, col)
    
    # Build neighbor map
    neighbors = {}
    for name, (row, col) in tile_to_grid.items():
        neighbors[name] = {
            'north': grid.get((row - 1, col)),  # Up = north = lower row
            'south': grid.get((row + 1, col)),  # Down = south = higher row
            'west': grid.get((row, col - 1)),   # Left = west
            'east': grid.get((row, col + 1))    # Right = east
        }
    
    return {
        'grid': grid,
        'tile_to_grid': tile_to_grid,
        'neighbors': neighbors,
        'cols': len(x_coords),
        'rows': len(y_coords)
    }


# =============================================================================
# INTERACTIVE HTML VIEWER
# =============================================================================

def create_html_viewer(tile_metadata, overview_info, grid_info, output_path, image_format):
    """
    Create interactive HTML viewer with:
    - Google Maps-style continuous scrolling (drag to pan across all tiles)
    - Dynamic tile loading as you pan
    - UTM to WGS84 coordinate conversion
    - Coordinate search (find tile by coordinates)
    - Brightness/contrast controls
    """
    print("\nGenerating interactive HTML viewer...")
    
    valid_tiles = [t for t in tile_metadata if 'error' not in t]
    if not valid_tiles or not overview_info:
        return None
    
    bounds = overview_info['bounds']
    img_size = overview_info['size']
    
    ext = 'jpg' if image_format in ['jpg', 'jpeg'] else 'png'
    
    # Detect CRS from tiles
    crs_str = valid_tiles[0].get('crs', '') if valid_tiles else ''
    print(f"Detected CRS: {crs_str}")
    
    # Determine projection parameters based on CRS or coordinate values
    crs_type = None
    
    # First try to detect from CRS string
    if '32634' in crs_str or 'UTM zone 34' in crs_str:
        crs_type = 'UTM34N'
    elif '32633' in crs_str or 'UTM zone 33' in crs_str:
        crs_type = 'UTM33N'
    elif '3794' in crs_str or 'D96' in crs_str:
        crs_type = 'D96TM'
    elif '3765' in crs_str or 'HTRS96' in crs_str:
        crs_type = 'HTRS96'
    elif '3912' in crs_str or 'D48' in crs_str or 'MGI' in crs_str:
        crs_type = 'D48GK'
    
    # If CRS not detected from metadata, infer from coordinate values
    if crs_type is None:
        sample_x = valid_tiles[0]['center']['x']
        sample_y = valid_tiles[0]['center']['y']
        print(f"Sample coordinates: E={sample_x:.1f}, N={sample_y:.1f}")
        
        # Croatian HTRS96/TM (EPSG:3765): E ~200,000-700,000, N ~4,700,000-5,200,000
        # Central meridian 16.5°E, scale factor 0.9999
        # Slovenia D96/TM (EPSG:3794): E ~370,000-630,000, N ~5,020,000-5,200,000
        # Central meridian 15.0°E, scale factor 0.9999
        
        # Croatian data typically has northing < 5,100,000 in southern areas
        # and easting varies widely across the country
        # Slovenian data typically has northing > 5,020,000
        
        # Check if likely Croatian (broader range, use HTRS96)
        if 100000 < sample_x < 750000 and 4650000 < sample_y < 5200000:
            # Could be either - check more carefully
            # Slovenian territory is roughly E: 370-630km, N: 5020-5200km
            # Croatian territory is roughly E: 150-700km, N: 4700-5150km
            if sample_x < 350000 or sample_x > 650000 or sample_y < 5000000:
                crs_type = 'HTRS96'
                print("Auto-detected: EPSG:3765 (HTRS96/TM Croatia) based on coordinates")
            else:
                # In overlapping range - default to Croatian since it's more common
                crs_type = 'HTRS96'
                print("Auto-detected: EPSG:3765 (HTRS96/TM Croatia) based on coordinates")
        elif 350000 < sample_x < 650000 and 5000000 < sample_y < 5250000:
            crs_type = 'D96TM'
            print("Auto-detected: EPSG:3794 (D96/TM Slovenia) based on coordinates")
        else:
            crs_type = 'HTRS96'
            print("Warning: Could not detect CRS, defaulting to EPSG:3765 (HTRS96/TM Croatia)")
    
    # Projection parameters
    # Note: D96/TM and D48/GK both use central meridian 15°E for zone 5
    # The difference is D48 uses Bessel ellipsoid, D96 uses GRS80/WGS84
    # For practical purposes with LiDAR viewing, D96TM params work well enough
    projection_params = {
        'HTRS96': {'central_meridian': 16.5, 'scale_factor': 0.9999, 'false_easting': 500000, 'ellipsoid': 'GRS80'},
        'D96TM': {'central_meridian': 15.0, 'scale_factor': 0.9999, 'false_easting': 500000, 'ellipsoid': 'GRS80'},
        'D48GK': {'central_meridian': 15.0, 'scale_factor': 1.0, 'false_easting': 500000, 'ellipsoid': 'BESSEL'},
        'UTM33N': {'central_meridian': 15.0, 'scale_factor': 0.9996, 'false_easting': 500000, 'ellipsoid': 'WGS84'},
        'UTM34N': {'central_meridian': 21.0, 'scale_factor': 0.9996, 'false_easting': 500000, 'ellipsoid': 'WGS84'},
    }
    
    proj_params = projection_params.get(crs_type, projection_params['D96TM'])
    print(f"Using projection: {crs_type} (central meridian: {proj_params['central_meridian']}°E)")
    
    # Build tiles JSON
    tiles_json = []
    for tile in valid_tiles:
        tiles_json.append({
            'name': tile['name'],
            'image': f"{tile['name']}.{ext}",
            'bounds': tile['bounds'],
            'center': tile['center'],
            'width': tile['width'],
            'height': tile['height'],
        })
    
    # Calculate world dimensions (total geographic extent)
    world_bounds = {
        'min_x': bounds['min_x'],
        'max_x': bounds['max_x'],
        'min_y': bounds['min_y'],
        'max_y': bounds['max_y']
    }
    
    html_content = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Archaeological Survey Viewer</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            overflow: hidden;
            height: 100vh;
        }}
        
        .app {{ display: flex; height: 100vh; }}
        
        /* Sidebar */
        .sidebar {{
            width: 320px;
            background: #111118;
            border-right: 1px solid #252530;
            display: flex;
            flex-direction: column;
            flex-shrink: 0;
            overflow: hidden;
        }}
        
        .sidebar-header {{
            padding: 12px 16px;
            border-bottom: 1px solid #252530;
        }}
        
        .sidebar-header h1 {{
            font-size: 1.1em;
            color: #4ecdc4;
            margin-bottom: 4px;
        }}
        
        .sidebar-header .subtitle {{
            font-size: 0.75em;
            color: #666;
        }}
        
        /* Control sections */
        .control-section {{
            padding: 12px 16px;
            border-bottom: 1px solid #252530;
        }}
        
        .control-section h3 {{
            font-size: 0.8em;
            color: #888;
            margin-bottom: 8px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}
        
        .search-box {{
            width: 100%;
            padding: 8px 10px;
            background: #1a1a24;
            border: 1px solid #252530;
            border-radius: 4px;
            color: #fff;
            font-size: 12px;
            margin-bottom: 8px;
        }}
        
        .search-box:focus {{
            outline: none;
            border-color: #4ecdc4;
        }}
        
        .search-box::placeholder {{
            color: #555;
        }}
        
        .search-container {{
            position: relative;
        }}
        
        .search-dropdown {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #1a1a24;
            border: 1px solid #4ecdc4;
            border-top: none;
            border-radius: 0 0 4px 4px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 100;
            display: none;
        }}
        
        .search-dropdown.visible {{
            display: block;
        }}
        
        .search-result {{
            padding: 8px 10px;
            font-size: 11px;
            font-family: monospace;
            cursor: pointer;
            border-bottom: 1px solid #252530;
            transition: background 0.1s;
        }}
        
        .search-result:last-child {{
            border-bottom: none;
        }}
        
        .search-result:hover,
        .search-result.selected {{
            background: #252530;
        }}
        
        .search-result .tile-name {{
            color: #4ecdc4;
        }}
        
        .search-result .tile-coords {{
            color: #666;
            font-size: 10px;
            margin-top: 2px;
        }}
        
        .coord-input-row {{
            display: flex;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .coord-input {{
            flex: 1;
            padding: 8px;
            background: #1a1a24;
            border: 1px solid #252530;
            border-radius: 4px;
            color: #fff;
            font-size: 11px;
            font-family: monospace;
        }}
        
        .coord-input:focus {{
            outline: none;
            border-color: #4ecdc4;
        }}
        
        .btn {{
            padding: 6px 12px;
            background: #252530;
            border: none;
            border-radius: 4px;
            color: #e0e0e0;
            font-size: 11px;
            cursor: pointer;
            transition: all 0.15s;
        }}
        
        .btn:hover {{ background: #353540; }}
        .btn.active {{ background: #4ecdc4; color: #000; }}
        .btn.primary {{ background: #4ecdc4; color: #000; }}
        .btn.primary:hover {{ background: #5fd9d0; }}
        
        .btn-row {{
            display: flex;
            gap: 6px;
            flex-wrap: wrap;
        }}
        
        /* Slider controls */
        .slider-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin-bottom: 8px;
        }}
        
        .slider-label {{
            font-size: 11px;
            color: #888;
            width: 70px;
        }}
        
        .slider {{
            flex: 1;
            accent-color: #4ecdc4;
        }}
        
        .slider-value {{
            font-size: 10px;
            font-family: monospace;
            color: #4ecdc4;
            width: 35px;
            text-align: right;
        }}
        
        /* Overview panel */
        .overview-section {{
            padding: 12px 16px;
            border-bottom: 1px solid #252530;
        }}
        
        .overview-container {{
            position: relative;
            cursor: crosshair;
            background: #0a0a0f;
            border-radius: 4px;
            overflow: hidden;
        }}
        
        .overview-image {{
            width: 100%;
            display: block;
        }}
        
        .overview-viewport {{
            position: absolute;
            border: 2px solid #4ecdc4;
            background: rgba(78, 205, 196, 0.15);
            pointer-events: none;
        }}
        
        /* Tile info */
        .tile-info {{
            padding: 12px 16px;
            font-size: 11px;
            flex: 1;
            overflow-y: auto;
        }}
        
        .tile-info h3 {{
            color: #4ecdc4;
            font-size: 12px;
            margin-bottom: 8px;
            word-break: break-all;
        }}
        
        .info-row {{
            display: flex;
            justify-content: space-between;
            padding: 4px 0;
            border-bottom: 1px solid #1a1a24;
        }}
        
        .info-label {{ color: #666; }}
        .info-value {{ 
            color: #fff; 
            font-family: monospace; 
            font-size: 10px;
            cursor: pointer;
        }}
        .info-value:hover {{ color: #4ecdc4; }}
        
        /* Tile list */
        .tile-list {{
            flex: 1;
            overflow-y: auto;
            padding: 8px;
        }}
        
        .tile-item {{
            padding: 6px 10px;
            margin: 2px 0;
            border-radius: 4px;
            cursor: pointer;
            font-size: 11px;
            font-family: monospace;
            transition: all 0.1s;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .tile-item:hover {{ background: #1a1a24; }}
        .tile-item.visible {{ background: #1a1a24; border-left: 2px solid #4ecdc4; }}
        
        /* Main map area */
        .main {{
            flex: 1;
            display: flex;
            flex-direction: column;
            overflow: hidden;
        }}
        
        .toolbar {{
            padding: 8px 12px;
            background: #111118;
            border-bottom: 1px solid #252530;
            display: flex;
            align-items: center;
            gap: 12px;
        }}
        
        .coord-display {{
            margin-left: auto;
            font-family: monospace;
            font-size: 11px;
            padding: 6px 10px;
            background: #1a1a24;
            border-radius: 4px;
            cursor: pointer;
            display: flex;
            gap: 16px;
        }}
        
        .coord-display:hover {{ background: #252530; }}
        .coord-display.copied {{ background: #4ecdc4; color: #000; }}
        
        .coord-utm {{ color: #4ecdc4; }}
        .coord-wgs {{ color: #f0a500; }}
        
        /* Map container */
        .map-container {{
            flex: 1;
            position: relative;
            overflow: hidden;
            background: #000;
            cursor: grab;
        }}
        
        .map-container:active {{ cursor: grabbing; }}
        
        .map-world {{
            position: absolute;
            transform-origin: 0 0;
        }}
        
        .map-tile {{
            position: absolute;
            background: #111;
            border: 1px solid #222;
        }}
        
        .map-tile img {{
            width: 100%;
            height: 100%;
            display: block;
            opacity: 0;
            transition: opacity 0.3s;
        }}
        
        .map-tile img.loaded {{ opacity: 1; }}
        
        .map-tile .tile-label {{
            position: absolute;
            bottom: 4px;
            left: 4px;
            font-size: 9px;
            font-family: monospace;
            color: #4ecdc4;
            background: rgba(0,0,0,0.7);
            padding: 2px 4px;
            border-radius: 2px;
            pointer-events: none;
            opacity: 0.7;
        }}
        
        /* Zoom indicator */
        .zoom-indicator {{
            position: absolute;
            bottom: 12px;
            right: 12px;
            background: rgba(0,0,0,0.8);
            padding: 6px 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            color: #4ecdc4;
        }}
        
        /* Loading indicator */
        .loading-tiles {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: rgba(0,0,0,0.8);
            padding: 6px 12px;
            border-radius: 4px;
            font-size: 11px;
            color: #f0a500;
            display: none;
        }}
        
        /* Help text - moved to toolbar area */
        .help-strip {{
            display: flex;
            align-items: center;
            gap: 12px;
            font-size: 10px;
            color: #666;
            padding: 6px 12px;
            background: #0d0d12;
            border-top: 1px solid #252530;
        }}
        
        .help-strip kbd {{
            background: #333;
            padding: 1px 4px;
            border-radius: 2px;
            font-family: monospace;
            color: #888;
        }}
        
        .help-strip .info-btn {{
            background: #252530;
            border: none;
            color: #4ecdc4;
            width: 20px;
            height: 20px;
            border-radius: 50%;
            cursor: pointer;
            font-size: 12px;
            font-weight: bold;
            display: flex;
            align-items: center;
            justify-content: center;
        }}
        
        .help-strip .info-btn:hover {{
            background: #4ecdc4;
            color: #000;
        }}
        
        /* Info modal */
        .info-modal {{
            display: none;
            position: fixed;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            background: rgba(0,0,0,0.8);
            z-index: 1000;
            align-items: center;
            justify-content: center;
        }}
        
        .info-modal.visible {{
            display: flex;
        }}
        
        .info-modal-content {{
            background: #111118;
            border: 1px solid #252530;
            border-radius: 8px;
            max-width: 700px;
            max-height: 80vh;
            overflow-y: auto;
            padding: 24px;
            position: relative;
        }}
        
        .info-modal-close {{
            position: absolute;
            top: 12px;
            right: 12px;
            background: none;
            border: none;
            color: #666;
            font-size: 24px;
            cursor: pointer;
        }}
        
        .info-modal-close:hover {{
            color: #fff;
        }}
        
        .info-modal h2 {{
            color: #4ecdc4;
            margin-bottom: 16px;
            font-size: 18px;
        }}
        
        .info-modal h3 {{
            color: #4ecdc4;
            margin: 20px 0 10px 0;
            font-size: 14px;
        }}
        
        .info-modal p {{
            color: #ccc;
            font-size: 13px;
            line-height: 1.6;
            margin-bottom: 12px;
        }}
        
        .info-modal code {{
            background: #1a1a24;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: monospace;
            color: #4ecdc4;
        }}
        
        .info-modal .formula {{
            background: #1a1a24;
            padding: 12px;
            border-radius: 4px;
            font-family: monospace;
            font-size: 12px;
            color: #ccc;
            margin: 12px 0;
            overflow-x: auto;
        }}
        
        /* Checkbox styling */
        .checkbox-row {{
            display: flex;
            align-items: center;
            gap: 8px;
            margin: 8px 0;
            font-size: 11px;
        }}
        
        .checkbox-row input[type="checkbox"] {{
            accent-color: #4ecdc4;
            width: 14px;
            height: 14px;
        }}
        
        /* Address search */
        .address-container {{
            position: relative;
        }}
        
        .address-dropdown {{
            position: absolute;
            top: 100%;
            left: 0;
            right: 0;
            background: #1a1a24;
            border: 1px solid #4ecdc4;
            border-top: none;
            border-radius: 0 0 4px 4px;
            max-height: 200px;
            overflow-y: auto;
            z-index: 100;
            display: none;
        }}
        
        .address-dropdown.visible {{
            display: block;
        }}
        
        .address-result {{
            padding: 8px 10px;
            font-size: 11px;
            cursor: pointer;
            border-bottom: 1px solid #252530;
            transition: background 0.1s;
        }}
        
        .address-result:hover {{
            background: #252530;
        }}
        
        .address-result .address-name {{
            color: #4ecdc4;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        .address-result .address-detail {{
            color: #666;
            font-size: 10px;
            margin-top: 2px;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }}
        
        /* Map base layer (satellite/street) */
        .map-base-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}
        
        .map-labels-layer {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
        }}
    </style>
</head>
<body>
    <div class="app">
        <div class="sidebar">
            <div class="sidebar-header">
                <h1>🔍 Archaeological Survey</h1>
                <div class="subtitle">LiDAR Terrain Viewer • {len(valid_tiles)} tiles</div>
            </div>
            
            <div class="control-section">
                <h3>Search Tiles</h3>
                <div class="search-container">
                    <input type="text" class="search-box" id="searchBox" placeholder="Tile name (use * wildcard)" autocomplete="off">
                    <div class="search-dropdown" id="searchDropdown"></div>
                </div>
            </div>
            
            <div class="control-section">
                <h3>Search Address</h3>
                <div class="address-container">
                    <input type="text" class="search-box" id="addressBox" placeholder="Street, city, or place name" autocomplete="off">
                    <div class="address-dropdown" id="addressDropdown"></div>
                </div>
            </div>
            
            <div class="control-section">
                <h3>Go to Coordinates</h3>
                <div class="coord-input-row">
                    <input type="text" class="coord-input" id="coordX" placeholder="X / Longitude">
                    <input type="text" class="coord-input" id="coordY" placeholder="Y / Latitude">
                </div>
                <div class="btn-row">
                    <button class="btn primary" onclick="goToCoordinates()">Go</button>
                    <button class="btn" onclick="pasteCoordinates()">Paste</button>
                    <span style="font-size:10px; color:#666; margin-left:8px;">UTM or WGS84</span>
                </div>
            </div>
            
            <div class="control-section">
                <h3>Map Overlay</h3>
                <div class="slider-row">
                    <span class="slider-label">Satellite</span>
                    <input type="range" class="slider" id="satelliteOpacity" min="0" max="100" value="0">
                    <span class="slider-value" id="satelliteOpacityVal">0%</span>
                </div>
                <div class="checkbox-row">
                    <input type="checkbox" id="showMapLabels">
                    <label for="showMapLabels">Show place names & streets</label>
                </div>
            </div>
            
            <div class="control-section">
                <h3>Display</h3>
                <div class="slider-row">
                    <span class="slider-label">Brightness</span>
                    <input type="range" class="slider" id="brightness" min="50" max="150" value="100">
                    <span class="slider-value" id="brightnessVal">100%</span>
                </div>
                <div class="slider-row">
                    <span class="slider-label">Contrast</span>
                    <input type="range" class="slider" id="contrast" min="50" max="200" value="100">
                    <span class="slider-value" id="contrastVal">100%</span>
                </div>
                <div class="btn-row" style="margin-top:8px;">
                    <button class="btn" id="invertBtn" onclick="toggleInvert()">Invert</button>
                    <button class="btn" onclick="resetFilters()">Reset</button>
                    <button class="btn" id="labelsBtn" onclick="toggleLabels()">Labels</button>
                </div>
            </div>
            
            <div class="overview-section">
                <h3>Overview</h3>
                <div class="overview-container" id="overviewContainer">
                    <img src="overview_map.png" class="overview-image" id="overviewImage">
                    <div class="overview-viewport" id="overviewViewport"></div>
                </div>
            </div>
            
            <div class="tile-info" id="tileInfo">
                <div style="color:#555; font-style:italic;">Hover over map to see coordinates</div>
            </div>
        </div>
        
        <div class="main">
            <div class="toolbar">
                <span style="font-size:11px; color:#888;">Zoom:</span>
                <button class="btn" onclick="setZoom(0.25)">25%</button>
                <button class="btn" onclick="setZoom(0.5)">50%</button>
                <button class="btn" onclick="setZoom(1)">100%</button>
                <button class="btn" onclick="setZoom(2)">200%</button>
                <button class="btn" onclick="fitAll()">Fit All</button>
                
                <div class="coord-display" id="coordDisplay" onclick="copyCoordinates()" title="Click to copy">
                    <span class="coord-utm">UTM: --</span>
                    <span class="coord-wgs">WGS84: --</span>
                </div>
            </div>
            
            <div class="map-container" id="mapContainer">
                <div class="map-base-layer" id="mapBaseLayer"></div>
                <div class="map-world" id="mapWorld"></div>
                <div class="map-labels-layer" id="mapLabelsLayer"></div>
                
                <div class="zoom-indicator" id="zoomIndicator">100%</div>
                <div class="loading-tiles" id="loadingIndicator">Loading tiles...</div>
            </div>
            
            <div class="help-strip">
                <span>Drag to pan</span>
                <span>Scroll to zoom</span>
                <span><kbd>C</kbd> copy WGS84</span>
                <span><kbd>M</kbd> copy HTRS96</span>
                <span><kbd>L</kbd> copy tile name</span>
                <button class="info-btn" onclick="showInfoModal()" title="Coordinate systems info">?</button>
            </div>
        </div>
    </div>
    
    <!-- Info Modal -->
    <div class="info-modal" id="infoModal">
        <div class="info-modal-content">
            <button class="info-modal-close" onclick="hideInfoModal()">&times;</button>
            
            <h2>Coordinate Systems Explained</h2>
            
            <h3>1. HTRS96/TM (EPSG:3765) - Croatian Projected Coordinates</h3>
            <p>Example: <code>E: 444,264.9  N: 5,111,092.4</code></p>
            <p>This is a <strong>Transverse Mercator projection</strong> used by Croatia.</p>
            <p><strong>Parameters:</strong></p>
            <p>
                • <strong>Ellipsoid:</strong> GRS80 (essentially identical to WGS84)<br>
                • <strong>Central meridian:</strong> 16.5°E (longitude where the projection is centered)<br>
                • <strong>False easting:</strong> 500,000m (added to all easting values to avoid negative numbers)<br>
                • <strong>False northing:</strong> 0m<br>
                • <strong>Scale factor:</strong> 0.9999 (slight reduction to minimize distortion)
            </p>
            <p><strong>How it works:</strong></p>
            <p>
                • <strong>Easting (E):</strong> Distance in meters from the central meridian, plus 500,000m. 
                E: 444,265 means the point is 55,735m west of the 16.5°E meridian.<br>
                • <strong>Northing (N):</strong> Distance in meters from the equator along the curved surface. 
                N: 5,111,092 means ~5,111km north of the equator.
            </p>
            <p><strong>Why use it?</strong> Meters are intuitive for measuring distances, works well for country-sized areas, 
            coordinates are always positive, and you can calculate distances easily with the Pythagorean theorem.</p>
            
            <h3>2. WGS84 (EPSG:4326) - Global Geographic Coordinates</h3>
            <p>Example: <code>46.137375°N, 15.778623°E</code></p>
            <p>This is a <strong>geographic coordinate system</strong> using latitude/longitude on an ellipsoid.</p>
            <p><strong>How it works:</strong></p>
            <p>
                • <strong>Latitude (46.137°N):</strong> Angle north of the equator.<br>
                • <strong>Longitude (15.778°E):</strong> Angle east of the Prime Meridian (Greenwich).
            </p>
            <p><strong>Why use it?</strong> Universal - works anywhere on Earth. Used by GPS, Google Maps, and most web mapping.</p>
            
            <h3>The Conversion</h3>
            <div class="formula">
Easting → Longitude:
  444,264.9 - 500,000 = -55,735.1m from central meridian
  At 46°N, 1° longitude ≈ 77.8km
  -55,735m ÷ 77,800m/° ≈ -0.716°
  16.5° + (-0.716°) ≈ 15.78°E ✓

Northing → Latitude:
  5,111,092.4m from equator
  (Complex calculation involving ellipsoid geometry)
  Result: 46.137°N ✓
            </div>
            <p>The math involves series expansions because the Earth is an ellipsoid, not a sphere, 
            so the relationship between meters and degrees varies with latitude.</p>
        </div>
    </div>

    <script>
        // =====================================================================
        // DATA
        // =====================================================================
        const tiles = {json.dumps(tiles_json)};
        const worldBounds = {json.dumps(world_bounds)};
        const overviewSize = {json.dumps(img_size)};
        const PROJ = {json.dumps(proj_params)};
        
        // =====================================================================
        // STATE
        // =====================================================================
        let currentZoom = 0.5;
        let panX = 0;
        let panY = 0;
        let isDragging = false;
        let dragStartX = 0;
        let dragStartY = 0;
        let panStartX = 0;
        let panStartY = 0;
        let isInverted = false;
        let showLabels = true;
        let lastMouseGeoX = null;
        let lastMouseGeoY = null;
        
        // Calculate world dimensions in pixels at zoom=1
        // Use average tile pixel density
        const sampleTile = tiles[0];
        const tileGeoWidth = sampleTile.bounds.right - sampleTile.bounds.left;
        const tileGeoHeight = sampleTile.bounds.top - sampleTile.bounds.bottom;
        const pixelsPerGeoUnit = sampleTile.width / tileGeoWidth;
        
        const worldGeoWidth = worldBounds.max_x - worldBounds.min_x;
        const worldGeoHeight = worldBounds.max_y - worldBounds.min_y;
        const worldPixelWidth = worldGeoWidth * pixelsPerGeoUnit;
        const worldPixelHeight = worldGeoHeight * pixelsPerGeoUnit;
        
        // Pre-calculate tile positions in world pixel coordinates
        tiles.forEach(t => {{
            t.worldX = (t.bounds.left - worldBounds.min_x) * pixelsPerGeoUnit;
            t.worldY = (worldBounds.max_y - t.bounds.top) * pixelsPerGeoUnit;
            t.worldW = (t.bounds.right - t.bounds.left) * pixelsPerGeoUnit;
            t.worldH = (t.bounds.top - t.bounds.bottom) * pixelsPerGeoUnit;
            t.loaded = false;
            t.element = null;
        }});
        
        // =====================================================================
        // TRANSVERSE MERCATOR <-> WGS84 CONVERSION
        // =====================================================================
        function tmToLatLng(easting, northing) {{
            // WGS84 ellipsoid parameters
            const a = 6378137.0;  // semi-major axis
            const f = 1 / 298.257223563;  // flattening
            const k0 = PROJ.scale_factor;
            const lng0 = PROJ.central_meridian * Math.PI / 180;
            const fe = PROJ.false_easting;
            
            const e2 = 2 * f - f * f;  // eccentricity squared
            const e = Math.sqrt(e2);
            const e4 = e2 * e2;
            const e6 = e4 * e2;
            const ep2 = e2 / (1 - e2);  // second eccentricity squared
            
            const x = easting - fe;
            const y = northing;
            
            // Footpoint latitude
            const M = y / k0;
            const mu = M / (a * (1 - e2/4 - 3*e4/64 - 5*e6/256));
            
            const e1 = (1 - Math.sqrt(1 - e2)) / (1 + Math.sqrt(1 - e2));
            const e12 = e1 * e1;
            const e13 = e12 * e1;
            const e14 = e13 * e1;
            
            const phi1 = mu 
                + (3*e1/2 - 27*e13/32) * Math.sin(2*mu)
                + (21*e12/16 - 55*e14/32) * Math.sin(4*mu)
                + (151*e13/96) * Math.sin(6*mu)
                + (1097*e14/512) * Math.sin(8*mu);
            
            const sinPhi1 = Math.sin(phi1);
            const cosPhi1 = Math.cos(phi1);
            const tanPhi1 = sinPhi1 / cosPhi1;
            
            const N1 = a / Math.sqrt(1 - e2 * sinPhi1 * sinPhi1);
            const T1 = tanPhi1 * tanPhi1;
            const C1 = ep2 * cosPhi1 * cosPhi1;
            const R1 = a * (1 - e2) / Math.pow(1 - e2 * sinPhi1 * sinPhi1, 1.5);
            const D = x / (N1 * k0);
            const D2 = D * D;
            const D3 = D2 * D;
            const D4 = D3 * D;
            const D5 = D4 * D;
            const D6 = D5 * D;
            
            const lat = phi1 - (N1 * tanPhi1 / R1) * (
                D2/2 
                - (5 + 3*T1 + 10*C1 - 4*C1*C1 - 9*ep2) * D4/24
                + (61 + 90*T1 + 298*C1 + 45*T1*T1 - 252*ep2 - 3*C1*C1) * D6/720
            );
            
            const lng = lng0 + (
                D 
                - (1 + 2*T1 + C1) * D3/6
                + (5 - 2*C1 + 28*T1 - 3*C1*C1 + 8*ep2 + 24*T1*T1) * D5/120
            ) / cosPhi1;
            
            return {{
                lat: lat * 180 / Math.PI,
                lng: lng * 180 / Math.PI
            }};
        }}
        
        function latLngToTm(lat, lng) {{
            const a = 6378137.0;
            const f = 1 / 298.257223563;
            const k0 = PROJ.scale_factor;
            const lng0 = PROJ.central_meridian * Math.PI / 180;
            const fe = PROJ.false_easting;
            
            const e2 = 2 * f - f * f;
            const ep2 = e2 / (1 - e2);
            
            const phi = lat * Math.PI / 180;
            const lambda = lng * Math.PI / 180;
            
            const sinPhi = Math.sin(phi);
            const cosPhi = Math.cos(phi);
            const tanPhi = sinPhi / cosPhi;
            
            const N = a / Math.sqrt(1 - e2 * sinPhi * sinPhi);
            const T = tanPhi * tanPhi;
            const C = ep2 * cosPhi * cosPhi;
            const A = cosPhi * (lambda - lng0);
            const A2 = A * A;
            const A3 = A2 * A;
            const A4 = A3 * A;
            const A5 = A4 * A;
            const A6 = A5 * A;
            
            const e4 = e2 * e2;
            const e6 = e4 * e2;
            const M = a * (
                (1 - e2/4 - 3*e4/64 - 5*e6/256) * phi
                - (3*e2/8 + 3*e4/32 + 45*e6/1024) * Math.sin(2*phi)
                + (15*e4/256 + 45*e6/1024) * Math.sin(4*phi)
                - (35*e6/3072) * Math.sin(6*phi)
            );
            
            const easting = fe + k0 * N * (
                A 
                + (1 - T + C) * A3/6
                + (5 - 18*T + T*T + 72*C - 58*ep2) * A5/120
            );
            
            const northing = k0 * (
                M + N * tanPhi * (
                    A2/2
                    + (5 - T + 9*C + 4*C*C) * A4/24
                    + (61 - 58*T + T*T + 600*C - 330*ep2) * A6/720
                )
            );
            
            return {{ easting, northing }};
        }}
        
        function formatLatLng(lat, lng) {{
            return lat.toFixed(6) + ', ' + lng.toFixed(6);
        }}
        
        // =====================================================================
        // MAP RENDERING
        // =====================================================================
        const mapContainer = document.getElementById('mapContainer');
        const mapWorld = document.getElementById('mapWorld');
        
        function updateMapTransform() {{
            mapWorld.style.transform = `translate(${{panX}}px, ${{panY}}px) scale(${{currentZoom}})`;
            updateVisibleTiles();
            updateOverviewViewport();
            updateMapLayers();
            document.getElementById('zoomIndicator').textContent = Math.round(currentZoom * 100) + '%';
        }}
        
        function updateVisibleTiles() {{
            const containerRect = mapContainer.getBoundingClientRect();
            const viewLeft = -panX / currentZoom;
            const viewTop = -panY / currentZoom;
            const viewRight = viewLeft + containerRect.width / currentZoom;
            const viewBottom = viewTop + containerRect.height / currentZoom;
            
            let loadingCount = 0;
            
            tiles.forEach(t => {{
                const visible = !(t.worldX + t.worldW < viewLeft || 
                                 t.worldX > viewRight ||
                                 t.worldY + t.worldH < viewTop || 
                                 t.worldY > viewBottom);
                
                if (visible && !t.element) {{
                    // Create tile element
                    const div = document.createElement('div');
                    div.className = 'map-tile';
                    div.style.left = t.worldX + 'px';
                    div.style.top = t.worldY + 'px';
                    div.style.width = t.worldW + 'px';
                    div.style.height = t.worldH + 'px';
                    
                    const img = document.createElement('img');
                    img.onload = () => {{
                        img.classList.add('loaded');
                        t.loaded = true;
                        updateLoadingIndicator();
                    }};
                    img.src = t.image;
                    div.appendChild(img);
                    
                    if (showLabels) {{
                        const label = document.createElement('div');
                        label.className = 'tile-label';
                        label.textContent = t.name;
                        div.appendChild(label);
                    }}
                    
                    mapWorld.appendChild(div);
                    t.element = div;
                    loadingCount++;
                }} else if (!visible && t.element) {{
                    // Remove tile element (%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%.%
                    // Keep loaded tiles in memory for smoother panning
                    // Only remove if far from viewport
                    const margin = 2000 / currentZoom;
                    const farAway = t.worldX + t.worldW < viewLeft - margin || 
                                   t.worldX > viewRight + margin ||
                                   t.worldY + t.worldH < viewTop - margin || 
                                   t.worldY > viewBottom + margin;
                    if (farAway) {{
                        t.element.remove();
                        t.element = null;
                        t.loaded = false;
                    }}
                }}
            }});
            
            updateLoadingIndicator();
            updateTileListVisibility(viewLeft, viewTop, viewRight, viewBottom);
        }}
        
        function updateLoadingIndicator() {{
            const loading = tiles.filter(t => t.element && !t.loaded).length;
            const indicator = document.getElementById('loadingIndicator');
            if (loading > 0) {{
                indicator.textContent = `Loading ${{loading}} tiles...`;
                indicator.style.display = 'block';
            }} else {{
                indicator.style.display = 'none';
            }}
        }}
        
        function updateTileListVisibility(viewLeft, viewTop, viewRight, viewBottom) {{
            // Highlight visible tiles in the list would go here if we add a tile list
        }}
        
        function updateOverviewViewport() {{
            const containerRect = mapContainer.getBoundingClientRect();
            const viewLeft = -panX / currentZoom;
            const viewTop = -panY / currentZoom;
            const viewWidth = containerRect.width / currentZoom;
            const viewHeight = containerRect.height / currentZoom;
            
            const overviewImg = document.getElementById('overviewImage');
            const viewport = document.getElementById('overviewViewport');
            const rect = overviewImg.getBoundingClientRect();
            
            const scaleX = rect.width / worldPixelWidth;
            const scaleY = rect.height / worldPixelHeight;
            
            viewport.style.left = (viewLeft * scaleX) + 'px';
            viewport.style.top = (viewTop * scaleY) + 'px';
            viewport.style.width = (viewWidth * scaleX) + 'px';
            viewport.style.height = (viewHeight * scaleY) + 'px';
        }}
        
        // =====================================================================
        // FILTERS
        // =====================================================================
        function updateFilters() {{
            const brightness = document.getElementById('brightness').value;
            const contrast = document.getElementById('contrast').value;
            
            document.getElementById('brightnessVal').textContent = brightness + '%';
            document.getElementById('contrastVal').textContent = contrast + '%';
            
            let filter = `brightness(${{brightness}}%) contrast(${{contrast}}%)`;
            if (isInverted) filter += ' invert(1)';
            
            mapWorld.style.filter = filter;
        }}
        
        function toggleInvert() {{
            isInverted = !isInverted;
            document.getElementById('invertBtn').classList.toggle('active', isInverted);
            updateFilters();
        }}
        
        function resetFilters() {{
            document.getElementById('brightness').value = 100;
            document.getElementById('contrast').value = 100;
            isInverted = false;
            document.getElementById('invertBtn').classList.remove('active');
            updateFilters();
        }}
        
        function toggleLabels() {{
            showLabels = !showLabels;
            document.getElementById('labelsBtn').classList.toggle('active', showLabels);
            document.querySelectorAll('.tile-label').forEach(l => {{
                l.style.display = showLabels ? 'block' : 'none';
            }});
        }}
        
        // =====================================================================
        // PAN & ZOOM
        // =====================================================================
        mapContainer.addEventListener('mousedown', e => {{
            isDragging = true;
            dragStartX = e.clientX;
            dragStartY = e.clientY;
            panStartX = panX;
            panStartY = panY;
            mapContainer.style.cursor = 'grabbing';
        }});
        
        window.addEventListener('mousemove', e => {{
            if (isDragging) {{
                panX = panStartX + (e.clientX - dragStartX);
                panY = panStartY + (e.clientY - dragStartY);
                updateMapTransform();
            }}
            
            // Update coordinates
            if (e.target.closest('.map-container')) {{
                const rect = mapContainer.getBoundingClientRect();
                const worldX = (e.clientX - rect.left - panX) / currentZoom;
                const worldY = (e.clientY - rect.top - panY) / currentZoom;
                
                const geoX = worldBounds.min_x + worldX / pixelsPerGeoUnit;
                const geoY = worldBounds.max_y - worldY / pixelsPerGeoUnit;
                
                lastMouseGeoX = geoX;
                lastMouseGeoY = geoY;
                
                const wgs = tmToLatLng(geoX, geoY);
                
                document.getElementById('coordDisplay').innerHTML = 
                    `<span class="coord-utm">E: ${{geoX.toFixed(1)}}, N: ${{geoY.toFixed(1)}}</span>` +
                    `<span class="coord-wgs">WGS84: ${{wgs.lat.toFixed(6)}}, ${{wgs.lng.toFixed(6)}}</span>`;
                
                // Update tile info
                const tile = tiles.find(t => 
                    geoX >= t.bounds.left && geoX <= t.bounds.right &&
                    geoY >= t.bounds.bottom && geoY <= t.bounds.top
                );
                if (tile) {{
                    currentTileName = tile.name;
                    const tileWgs = tmToLatLng(tile.center.x, tile.center.y);
                    document.getElementById('tileInfo').innerHTML = `
                        <h3>${{tile.name}}</h3>
                        <div class="info-row">
                            <span class="info-label">Size:</span>
                            <span class="info-value">${{tile.width}} × ${{tile.height}} px</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Easting:</span>
                            <span class="info-value" onclick="copyText('${{tile.bounds.left.toFixed(1)}}, ${{tile.bounds.right.toFixed(1)}}')">${{tile.bounds.left.toFixed(1)}} → ${{tile.bounds.right.toFixed(1)}}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Northing:</span>
                            <span class="info-value" onclick="copyText('${{tile.bounds.bottom.toFixed(1)}}, ${{tile.bounds.top.toFixed(1)}}')">${{tile.bounds.bottom.toFixed(1)}} → ${{tile.bounds.top.toFixed(1)}}</span>
                        </div>
                        <div class="info-row">
                            <span class="info-label">Center WGS84:</span>
                            <span class="info-value" onclick="copyText('${{tileWgs.lat.toFixed(6)}}, ${{tileWgs.lng.toFixed(6)}}')">${{formatLatLng(tileWgs.lat, tileWgs.lng)}}</span>
                        </div>
                    `;
                }} else {{
                    currentTileName = null;
                }}
            }}
        }});
        
        window.addEventListener('mouseup', () => {{
            isDragging = false;
            mapContainer.style.cursor = 'grab';
        }});
        
        mapContainer.addEventListener('wheel', e => {{
            e.preventDefault();
            
            const rect = mapContainer.getBoundingClientRect();
            const mouseX = e.clientX - rect.left;
            const mouseY = e.clientY - rect.top;
            
            const worldX = (mouseX - panX) / currentZoom;
            const worldY = (mouseY - panY) / currentZoom;
            
            const delta = e.deltaY > 0 ? 0.85 : 1.18;
            const newZoom = Math.max(0.05, Math.min(4, currentZoom * delta));
            
            // Zoom toward mouse position
            panX = mouseX - worldX * newZoom;
            panY = mouseY - worldY * newZoom;
            currentZoom = newZoom;
            
            updateMapTransform();
        }}, {{ passive: false }});
        
        function setZoom(z) {{
            const rect = mapContainer.getBoundingClientRect();
            const centerX = rect.width / 2;
            const centerY = rect.height / 2;
            
            const worldX = (centerX - panX) / currentZoom;
            const worldY = (centerY - panY) / currentZoom;
            
            currentZoom = z;
            
            panX = centerX - worldX * currentZoom;
            panY = centerY - worldY * currentZoom;
            
            updateMapTransform();
        }}
        
        function fitAll() {{
            const rect = mapContainer.getBoundingClientRect();
            const scaleX = rect.width / worldPixelWidth;
            const scaleY = rect.height / worldPixelHeight;
            currentZoom = Math.min(scaleX, scaleY) * 0.95;
            
            panX = (rect.width - worldPixelWidth * currentZoom) / 2;
            panY = (rect.height - worldPixelHeight * currentZoom) / 2;
            
            updateMapTransform();
        }}
        
        // =====================================================================
        // COORDINATE SEARCH
        // =====================================================================
        function goToCoordinates() {{
            let x = document.getElementById('coordX').value.trim();
            let y = document.getElementById('coordY').value.trim();
            
            if (!x || !y) return;
            
            x = parseFloat(x);
            y = parseFloat(y);
            
            if (isNaN(x) || isNaN(y)) return;
            
            // Detect if WGS84 (lat/lng) or projected coordinates
            // WGS84 lat is typically -90 to 90, lng -180 to 180
            // Projected easting is typically 100000-900000, northing 0-10000000
            let geoX, geoY;
            
            if (Math.abs(x) <= 180 && Math.abs(y) <= 90) {{
                // Looks like WGS84 (lng, lat) or (lat, lng)
                // Assume lat, lng order (Google Maps style)
                const tm = latLngToTm(x, y);
                geoX = tm.easting;
                geoY = tm.northing;
            }} else if (Math.abs(y) <= 180 && Math.abs(x) <= 90) {{
                // Swapped: x is lat, y is lng
                const tm = latLngToTm(y, x);
                geoX = tm.easting;
                geoY = tm.northing;
            }} else {{
                // Assume projected coordinates (easting, northing)
                geoX = x;
                geoY = y;
            }}
            
            // Check if within bounds
            if (geoX < worldBounds.min_x || geoX > worldBounds.max_x ||
                geoY < worldBounds.min_y || geoY > worldBounds.max_y) {{
                alert('Coordinates are outside the mapped area');
                return;
            }}
            
            // Pan to location
            const worldX = (geoX - worldBounds.min_x) * pixelsPerGeoUnit;
            const worldY = (worldBounds.max_y - geoY) * pixelsPerGeoUnit;
            
            const rect = mapContainer.getBoundingClientRect();
            
            // Set reasonable zoom
            if (currentZoom < 0.5) currentZoom = 0.5;
            
            panX = rect.width / 2 - worldX * currentZoom;
            panY = rect.height / 2 - worldY * currentZoom;
            
            updateMapTransform();
        }}
        
        function pasteCoordinates() {{
            navigator.clipboard.readText().then(text => {{
                // Try to parse coordinates from text
                // Support formats: "lat, lng" or "lat lng" or "x, y"
                const parts = text.split(/[,\\s]+/).filter(p => p.length > 0);
                if (parts.length >= 2) {{
                    document.getElementById('coordX').value = parts[0];
                    document.getElementById('coordY').value = parts[1];
                }}
            }}).catch(() => {{}});
        }}
        
        // =====================================================================
        // OVERVIEW CLICK
        // =====================================================================
        document.getElementById('overviewContainer').addEventListener('click', e => {{
            const img = document.getElementById('overviewImage');
            const rect = img.getBoundingClientRect();
            
            const clickX = (e.clientX - rect.left) / rect.width;
            const clickY = (e.clientY - rect.top) / rect.height;
            
            const worldX = clickX * worldPixelWidth;
            const worldY = clickY * worldPixelHeight;
            
            const containerRect = mapContainer.getBoundingClientRect();
            
            panX = containerRect.width / 2 - worldX * currentZoom;
            panY = containerRect.height / 2 - worldY * currentZoom;
            
            updateMapTransform();
        }});
        
        // =====================================================================
        // COPY COORDINATES
        // =====================================================================
        function copyCoordinates() {{
            if (lastMouseGeoX === null) return;
            
            const wgs = tmToLatLng(lastMouseGeoX, lastMouseGeoY);
            const text = `${{wgs.lat.toFixed(6)}}, ${{wgs.lng.toFixed(6)}}`;
            
            navigator.clipboard.writeText(text).then(() => {{
                const display = document.getElementById('coordDisplay');
                display.classList.add('copied');
                setTimeout(() => display.classList.remove('copied'), 500);
            }});
        }}
        
        function copyProjectedCoordinates() {{
            if (lastMouseGeoX === null) return;
            
            const text = `E: ${{lastMouseGeoX.toFixed(1)}}, N: ${{lastMouseGeoY.toFixed(1)}}`;
            
            navigator.clipboard.writeText(text).then(() => {{
                const display = document.getElementById('coordDisplay');
                display.classList.add('copied');
                setTimeout(() => display.classList.remove('copied'), 500);
            }});
        }}
        
        let currentTileName = null;
        
        function copyTileName() {{
            if (!currentTileName) return;
            
            navigator.clipboard.writeText(currentTileName).then(() => {{
                const display = document.getElementById('coordDisplay');
                display.classList.add('copied');
                setTimeout(() => display.classList.remove('copied'), 500);
            }});
        }}
        
        // =====================================================================
        // INFO MODAL
        // =====================================================================
        function showInfoModal() {{
            document.getElementById('infoModal').classList.add('visible');
        }}
        
        function hideInfoModal() {{
            document.getElementById('infoModal').classList.remove('visible');
        }}
        
        document.getElementById('infoModal').addEventListener('click', e => {{
            if (e.target.id === 'infoModal') hideInfoModal();
        }});
        
        document.addEventListener('keydown', e => {{
            if (e.key === 'Escape' && document.getElementById('infoModal').classList.contains('visible')) {{
                hideInfoModal();
            }}
        }});
        
        function copyText(text) {{
            navigator.clipboard.writeText(text);
        }}
        
        // =====================================================================
        // TILE SEARCH WITH DROPDOWN
        // =====================================================================
        const searchBox = document.getElementById('searchBox');
        const searchDropdown = document.getElementById('searchDropdown');
        let selectedResultIndex = -1;
        let currentResults = [];
        
        function updateSearchDropdown(query) {{
            if (!query) {{
                searchDropdown.classList.remove('visible');
                currentResults = [];
                selectedResultIndex = -1;
                return;
            }}
            
            // Convert * to regex
            const pattern = query.replace(/\\*/g, '.*');
            let regex;
            try {{
                regex = new RegExp(pattern, 'i');
            }} catch(e) {{
                return;
            }}
            
            // Find matching tiles (limit to 20 results)
            currentResults = tiles.filter(t => regex.test(t.name)).slice(0, 20);
            
            if (currentResults.length === 0) {{
                searchDropdown.innerHTML = '<div class="search-result" style="color:#666; cursor:default;">No matches found</div>';
                searchDropdown.classList.add('visible');
                return;
            }}
            
            // Build dropdown HTML
            searchDropdown.innerHTML = currentResults.map((t, idx) => {{
                const wgs = tmToLatLng(t.center.x, t.center.y);
                return `<div class="search-result${{idx === selectedResultIndex ? ' selected' : ''}}" data-index="${{idx}}">
                    <div class="tile-name">${{t.name}}</div>
                    <div class="tile-coords">${{wgs.lat.toFixed(4)}}, ${{wgs.lng.toFixed(4)}}</div>
                </div>`;
            }}).join('');
            
            searchDropdown.classList.add('visible');
            
            // Add click handlers
            searchDropdown.querySelectorAll('.search-result').forEach(el => {{
                el.addEventListener('click', () => {{
                    const idx = parseInt(el.dataset.index);
                    if (!isNaN(idx) && currentResults[idx]) {{
                        goToTile(currentResults[idx]);
                        searchDropdown.classList.remove('visible');
                        searchBox.value = currentResults[idx].name;
                    }}
                }});
            }});
        }}
        
        function goToTile(tile) {{
            const worldX = tile.worldX + tile.worldW / 2;
            const worldY = tile.worldY + tile.worldH / 2;
            
            const rect = mapContainer.getBoundingClientRect();
            if (currentZoom < 0.5) currentZoom = 0.5;
            
            panX = rect.width / 2 - worldX * currentZoom;
            panY = rect.height / 2 - worldY * currentZoom;
            
            updateMapTransform();
        }}
        
        searchBox.addEventListener('input', e => {{
            selectedResultIndex = -1;
            updateSearchDropdown(e.target.value.trim());
        }});
        
        searchBox.addEventListener('keydown', e => {{
            if (!searchDropdown.classList.contains('visible') || currentResults.length === 0) return;
            
            if (e.key === 'ArrowDown') {{
                e.preventDefault();
                selectedResultIndex = Math.min(selectedResultIndex + 1, currentResults.length - 1);
                updateSearchDropdown(searchBox.value.trim());
            }} else if (e.key === 'ArrowUp') {{
                e.preventDefault();
                selectedResultIndex = Math.max(selectedResultIndex - 1, 0);
                updateSearchDropdown(searchBox.value.trim());
            }} else if (e.key === 'Enter') {{
                e.preventDefault();
                if (selectedResultIndex >= 0 && currentResults[selectedResultIndex]) {{
                    goToTile(currentResults[selectedResultIndex]);
                    searchDropdown.classList.remove('visible');
                    searchBox.value = currentResults[selectedResultIndex].name;
                }} else if (currentResults.length > 0) {{
                    goToTile(currentResults[0]);
                    searchDropdown.classList.remove('visible');
                    searchBox.value = currentResults[0].name;
                }}
            }} else if (e.key === 'Escape') {{
                searchDropdown.classList.remove('visible');
            }}
        }});
        
        // Close dropdown when clicking outside
        document.addEventListener('click', e => {{
            if (!e.target.closest('.search-container')) {{
                searchDropdown.classList.remove('visible');
            }}
        }});
        
        searchBox.addEventListener('focus', () => {{
            if (searchBox.value.trim()) {{
                updateSearchDropdown(searchBox.value.trim());
            }}
        }});
        
        // =====================================================================
        // KEYBOARD
        // =====================================================================
        document.addEventListener('keydown', e => {{
            if (e.target.tagName === 'INPUT') return;
            
            switch(e.key) {{
                case 'c':
                    copyCoordinates();
                    break;
                case 'm':
                    copyProjectedCoordinates();
                    break;
                case 'l':
                    copyTileName();
                    break;
                case '+':
                case '=':
                    setZoom(Math.min(4, currentZoom * 1.25));
                    break;
                case '-':
                    setZoom(Math.max(0.05, currentZoom * 0.8));
                    break;
                case '0':
                    fitAll();
                    break;
                case 'i':
                    toggleInvert();
                    break;
            }}
        }});
        
        // =====================================================================
        // MAP OVERLAY LAYERS (OpenStreetMap)
        // =====================================================================
        const mapBaseLayer = document.getElementById('mapBaseLayer');
        const mapLabelsLayer = document.getElementById('mapLabelsLayer');
        let satelliteOpacity = 0;
        let showMapLabelsEnabled = false;
        
        // Tile URL templates
        const SATELLITE_URL = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/';
        const LABELS_URL = 'https://cartodb-basemaps-a.global.ssl.fastly.net/light_only_labels/';
        
        // Pre-calculate the WGS84 bounds of our data
        const wgsBounds = {{
            nw: tmToLatLng(worldBounds.min_x, worldBounds.max_y),
            ne: tmToLatLng(worldBounds.max_x, worldBounds.max_y),
            se: tmToLatLng(worldBounds.max_x, worldBounds.min_y),
            sw: tmToLatLng(worldBounds.min_x, worldBounds.min_y)
        }};
        const wgsCenter = tmToLatLng(
            (worldBounds.min_x + worldBounds.max_x) / 2,
            (worldBounds.min_y + worldBounds.max_y) / 2
        );
        
        function latLngToTileCoords(lat, lng, zoom) {{
            const n = Math.pow(2, zoom);
            const x = (lng + 180) / 360 * n;
            const latRad = lat * Math.PI / 180;
            const y = (1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * n;
            return {{ x, y }};
        }}
        
        function tileToLatLng(tileX, tileY, zoom) {{
            const n = Math.pow(2, zoom);
            const lng = tileX / n * 360 - 180;
            const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * tileY / n)));
            const lat = latRad * 180 / Math.PI;
            return {{ lat, lng }};
        }}
        
        function wgsToScreen(lat, lng) {{
            // Convert WGS84 to projected coordinates, then to screen
            const proj = latLngToTm(lat, lng);
            const worldX = (proj.easting - worldBounds.min_x) * pixelsPerGeoUnit;
            const worldY = (worldBounds.max_y - proj.northing) * pixelsPerGeoUnit;
            return {{
                x: worldX * currentZoom + panX,
                y: worldY * currentZoom + panY
            }};
        }}
        
        function updateMapLayers() {{
            if (satelliteOpacity === 0 && !showMapLabelsEnabled) {{
                mapBaseLayer.innerHTML = '';
                mapLabelsLayer.innerHTML = '';
                return;
            }}
            
            const containerRect = mapContainer.getBoundingClientRect();
            
            // Get the four corners of the screen in WGS84
            function screenToWgs(sx, sy) {{
                const worldX = (sx - panX) / currentZoom;
                const worldY = (sy - panY) / currentZoom;
                const geoX = worldBounds.min_x + worldX / pixelsPerGeoUnit;
                const geoY = worldBounds.max_y - worldY / pixelsPerGeoUnit;
                return tmToLatLng(geoX, geoY);
            }}
            
            const corners = [
                screenToWgs(0, 0),
                screenToWgs(containerRect.width, 0),
                screenToWgs(containerRect.width, containerRect.height),
                screenToWgs(0, containerRect.height)
            ];
            
            // Get bounding box in WGS84
            const lats = corners.map(c => c.lat);
            const lngs = corners.map(c => c.lng);
            const minLat = Math.min(...lats);
            const maxLat = Math.max(...lats);
            const minLng = Math.min(...lngs);
            const maxLng = Math.max(...lngs);
            
            // Determine appropriate tile zoom based on current view scale
            // Calculate meters per pixel at current zoom
            const metersPerPixel = 1 / (pixelsPerGeoUnit * currentZoom);
            // Standard web map: zoom 0 = 156543 m/px at equator, halves each level
            let tileZoom = Math.round(Math.log2(156543 / metersPerPixel));
            tileZoom = Math.max(8, Math.min(18, tileZoom));
            
            // Get tile coordinates for the visible area
            const topLeft = latLngToTileCoords(maxLat, minLng, tileZoom);
            const bottomRight = latLngToTileCoords(minLat, maxLng, tileZoom);
            
            const minTileX = Math.floor(topLeft.x) - 1;
            const maxTileX = Math.ceil(bottomRight.x) + 1;
            const minTileY = Math.floor(topLeft.y) - 1;
            const maxTileY = Math.ceil(bottomRight.y) + 1;
            
            // Clear and rebuild layers
            mapBaseLayer.innerHTML = '';
            mapLabelsLayer.innerHTML = '';
            
            for (let tx = minTileX; tx <= maxTileX; tx++) {{
                for (let ty = minTileY; ty <= maxTileY; ty++) {{
                    // Get tile corners in WGS84
                    const tileNW = tileToLatLng(tx, ty, tileZoom);
                    const tileSE = tileToLatLng(tx + 1, ty + 1, tileZoom);
                    
                    // Convert to screen coordinates
                    const nwScreen = wgsToScreen(tileNW.lat, tileNW.lng);
                    const seScreen = wgsToScreen(tileSE.lat, tileSE.lng);
                    
                    const left = nwScreen.x;
                    const top = nwScreen.y;
                    const width = seScreen.x - nwScreen.x;
                    const height = seScreen.y - nwScreen.y;
                    
                    // Skip if completely outside viewport or invalid size
                    if (width <= 0 || height <= 0) continue;
                    if (left + width < -500 || left > containerRect.width + 500 ||
                        top + height < -500 || top > containerRect.height + 500) {{
                        continue;
                    }}
                    
                    const style = `position:absolute; left:${{left.toFixed(1)}}px; top:${{top.toFixed(1)}}px; width:${{width.toFixed(1)}}px; height:${{height.toFixed(1)}}px;`;
                    
                    // Satellite layer
                    if (satelliteOpacity > 0) {{
                        const img = document.createElement('img');
                        img.src = SATELLITE_URL + tileZoom + '/' + ty + '/' + tx;
                        img.style.cssText = style;
                        img.crossOrigin = 'anonymous';
                        img.onerror = () => img.style.display = 'none';
                        mapBaseLayer.appendChild(img);
                    }}
                    
                    // Labels layer (only shows text labels, transparent background)
                    if (showMapLabelsEnabled) {{
                        const img = document.createElement('img');
                        img.src = LABELS_URL + tileZoom + '/' + tx + '/' + ty + '.png';
                        img.style.cssText = style;
                        img.crossOrigin = 'anonymous';
                        img.onerror = () => img.style.display = 'none';
                        mapLabelsLayer.appendChild(img);
                    }}
                }}
            }}
            
            // Satellite layer is always full opacity when visible
            // The slider controls how transparent the LiDAR layer becomes
            mapBaseLayer.style.opacity = satelliteOpacity > 0 ? 1 : 0;
            mapLabelsLayer.style.opacity = showMapLabelsEnabled ? 1 : 0;
            
            // Make LiDAR layer transparent based on satellite slider
            mapWorld.style.opacity = 1 - (satelliteOpacity / 100);
        }}
        
        document.getElementById('satelliteOpacity').addEventListener('input', e => {{
            satelliteOpacity = parseInt(e.target.value);
            document.getElementById('satelliteOpacityVal').textContent = satelliteOpacity + '%';
            updateMapLayers();
        }});
        
        document.getElementById('showMapLabels').addEventListener('change', e => {{
            showMapLabelsEnabled = e.target.checked;
            updateMapLayers();
        }});
        
        // =====================================================================
        // ADDRESS SEARCH (Nominatim/OpenStreetMap)
        // =====================================================================
        const addressBox = document.getElementById('addressBox');
        const addressDropdown = document.getElementById('addressDropdown');
        let addressSearchTimeout = null;
        
        function searchAddress(query) {{
            if (!query || query.length < 2) {{
                addressDropdown.classList.remove('visible');
                return;
            }}
            
            // Get center of our data for biasing results
            const centerX = (worldBounds.min_x + worldBounds.max_x) / 2;
            const centerY = (worldBounds.min_y + worldBounds.max_y) / 2;
            const centerWgs = tmToLatLng(centerX, centerY);
            
            // Use Photon API (Komoot) - better autocomplete support than Nominatim
            // Falls back to Nominatim if Photon fails
            const photonUrl = `https://photon.komoot.io/api/?q=${{encodeURIComponent(query)}}&lat=${{centerWgs.lat}}&lon=${{centerWgs.lng}}&limit=10&lang=en`;
            
            fetch(photonUrl)
            .then(r => r.json())
            .then(data => {{
                const results = data.features || [];
                if (results.length === 0) {{
                    // Fallback to Nominatim
                    return fetch(`https://nominatim.openstreetmap.org/search?format=json&q=${{encodeURIComponent(query)}}&limit=10&countrycodes=hr,si`, {{
                        headers: {{ 'User-Agent': 'LiDAR-Archaeological-Viewer/1.0' }}
                    }}).then(r => r.json()).then(nomResults => {{
                        return nomResults.map(r => ({{
                            name: r.display_name.split(',')[0],
                            detail: r.display_name.split(',').slice(1, 4).join(','),
                            lat: parseFloat(r.lat),
                            lng: parseFloat(r.lon)
                        }}));
                    }});
                }}
                return results.map(f => ({{
                    name: f.properties.name || f.properties.street || 'Unknown',
                    detail: [f.properties.city, f.properties.county, f.properties.country].filter(Boolean).join(', '),
                    lat: f.geometry.coordinates[1],
                    lng: f.geometry.coordinates[0]
                }}));
            }})
            .then(results => {{
                if (!results || results.length === 0) {{
                    addressDropdown.innerHTML = '<div class="address-result" style="color:#666; cursor:default;">No results found</div>';
                }} else {{
                    addressDropdown.innerHTML = results.map((r, idx) => `
                        <div class="address-result" data-lat="${{r.lat}}" data-lng="${{r.lng}}">
                            <div class="address-name">${{r.name}}</div>
                            <div class="address-detail">${{r.detail}}</div>
                        </div>
                    `).join('');
                    
                    addressDropdown.querySelectorAll('.address-result').forEach(el => {{
                        el.addEventListener('click', () => {{
                            const lat = parseFloat(el.dataset.lat);
                            const lng = parseFloat(el.dataset.lng);
                            if (!isNaN(lat) && !isNaN(lng)) {{
                                goToLatLng(lat, lng);
                                addressDropdown.classList.remove('visible');
                                addressBox.value = el.querySelector('.address-name').textContent;
                            }}
                        }});
                    }});
                }}
                addressDropdown.classList.add('visible');
            }})
            .catch(err => {{
                console.error('Address search error:', err);
                addressDropdown.innerHTML = '<div class="address-result" style="color:#666; cursor:default;">Search error - try again</div>';
                addressDropdown.classList.add('visible');
            }});
        }}
        
        function goToLatLng(lat, lng) {{
            const tm = latLngToTm(lat, lng);
            const geoX = tm.easting;
            const geoY = tm.northing;
            
            // Check if within bounds (with generous margin)
            const margin = 20000; // 20km margin
            const isOutside = geoX < worldBounds.min_x - margin || geoX > worldBounds.max_x + margin ||
                              geoY < worldBounds.min_y - margin || geoY > worldBounds.max_y + margin;
            
            if (isOutside) {{
                // Still allow navigation but warn
                console.log('Location may be outside mapped area');
            }}
            
            const worldX = (geoX - worldBounds.min_x) * pixelsPerGeoUnit;
            const worldY = (worldBounds.max_y - geoY) * pixelsPerGeoUnit;
            
            const rect = mapContainer.getBoundingClientRect();
            if (currentZoom < 0.3) currentZoom = 0.3;
            
            panX = rect.width / 2 - worldX * currentZoom;
            panY = rect.height / 2 - worldY * currentZoom;
            
            updateMapTransform();
        }}
        
        addressBox.addEventListener('input', e => {{
            clearTimeout(addressSearchTimeout);
            addressSearchTimeout = setTimeout(() => {{
                searchAddress(e.target.value.trim());
            }}, 400);
        }});
        
        addressBox.addEventListener('keydown', e => {{
            if (e.key === 'Escape') {{
                addressDropdown.classList.remove('visible');
            }}
        }});
        
        document.addEventListener('click', e => {{
            if (!e.target.closest('.address-container')) {{
                addressDropdown.classList.remove('visible');
            }}
        }});
        
        // =====================================================================
        // INIT
        // =====================================================================
        document.getElementById('brightness').addEventListener('input', updateFilters);
        document.getElementById('contrast').addEventListener('input', updateFilters);
        
        // Initial view
        fitAll();
        
        // Handle Enter key in coordinate inputs
        document.getElementById('coordX').addEventListener('keypress', e => {{
            if (e.key === 'Enter') goToCoordinates();
        }});
        document.getElementById('coordY').addEventListener('keypress', e => {{
            if (e.key === 'Enter') goToCoordinates();
        }});
    </script>
</body>
</html>'''
    
    html_path = output_path / 'viewer.html'
    with open(html_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Interactive viewer saved: {html_path}")
    return str(html_path)


# =============================================================================
# METADATA EXPORT
# =============================================================================

# =============================================================================
# METADATA EXPORT
# =============================================================================

def create_metadata_csv(tile_metadata, output_path):
    """Export tile metadata to CSV."""
    csv_path = output_path / 'tiles.csv'
    
    with open(csv_path, 'w', encoding='utf-8') as f:
        f.write('name,left,right,bottom,top,center_x,center_y,width_px,height_px,crs\n')
        
        for t in tile_metadata:
            if 'error' not in t:
                f.write(f"{t['name']},{t['bounds']['left']},{t['bounds']['right']},"
                       f"{t['bounds']['bottom']},{t['bounds']['top']},"
                       f"{t['center']['x']},{t['center']['y']},"
                       f"{t['width']},{t['height']},{t.get('crs', '')}\n")
    
    print(f"Metadata CSV saved: {csv_path}")


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Convert DMR TIF files to archaeological survey images',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
VERSION 3 - ARCHAEOLOGICAL DISCOVERY TOOL

This tool converts LiDAR/DMR terrain data into high-resolution images
optimized for detecting archaeological features.

Examples:
    # Standard usage (archaeological mode, full resolution)
    python convert_dmr_to_images.py -i ./tifs -o ./output
    
    # Colored terrain mode
    python convert_dmr_to_images.py -i ./tifs -o ./output --mode standard
    
    # Higher vertical exaggeration for very flat terrain
    python convert_dmr_to_images.py -i ./tifs -o ./output --z-factor 5

Visualization modes:
    archaeological - Grayscale, multi-directional hillshade + local relief
                     model. Best for detecting subtle features. (DEFAULT)
    standard       - Colored terrain with hillshade overlay.

After conversion, open viewer.html in your browser for:
    - Click-to-view on overview map
    - Full resolution pan/zoom on tiles
    - Geographic navigation (↑↓←→ = N/S/E/W neighbors)
    - Real-time brightness/contrast/invert controls
    - Coordinate tracking
"""
    )
    
    parser.add_argument('-i', '--input', required=True, help='Input directory with TIF files')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('-f', '--format', choices=['png', 'jpg'], default='png',
                        help='Output format (default: png)')
    parser.add_argument('-q', '--quality', type=int, default=98,
                        help='JPEG quality (default: 98)')
    parser.add_argument('-m', '--mode', choices=['archaeological', 'standard'], 
                        default='archaeological', help='Visualization mode (default: archaeological)')
    parser.add_argument('-c', '--colormap', default='terrain',
                        help='Colormap for standard mode (default: terrain)')
    parser.add_argument('-z', '--z-factor', type=float, default=3.0,
                        help='Vertical exaggeration (default: 3.0)')
    parser.add_argument('-w', '--workers', type=int, default=4,
                        help='Parallel workers (default: 4)')
    parser.add_argument('--reference-only', action='store_true',
                        help='Only generate overview/viewer, skip tile conversion')
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    output_path = Path(args.output)
    
    if not input_path.exists():
        print(f"Error: Input directory not found: {args.input}")
        sys.exit(1)
    
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Find TIFs
    tif_files = find_tif_files(input_path)
    if not tif_files:
        print(f"No TIF files found in {args.input}")
        sys.exit(1)
    
    print("=" * 60)
    print("DMR ARCHAEOLOGICAL SURVEY CONVERTER - v3")
    print("=" * 60)
    print(f"Input:  {input_path.absolute()}")
    print(f"Output: {output_path.absolute()}")
    print(f"Tiles:  {len(tif_files)}")
    print(f"Mode:   {args.mode}")
    print(f"Format: {args.format.upper()}")
    print(f"Z-factor: {args.z_factor}")
    print()
    
    # Collect metadata
    print("Reading tile metadata...")
    tile_metadata = []
    for f in tqdm(tif_files, desc="Scanning"):
        tile_metadata.append(get_tif_metadata(f))
    
    valid = [t for t in tile_metadata if 'error' not in t]
    errors = [t for t in tile_metadata if 'error' in t]
    
    if errors:
        print(f"Warning: {len(errors)} files had errors")
    
    print(f"Valid tiles: {len(valid)}")
    
    # Convert tiles
    if not args.reference_only:
        print(f"\nConverting tiles at FULL RESOLUTION...")
        
        ext = args.format
        conversion_args = [
            (Path(t['path']), output_path / f"{t['name']}.{ext}", 
             args.format, args.mode, args.colormap, args.z_factor, args.quality)
            for t in valid
        ]
        
        success = 0
        fail = 0
        
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(convert_single_tif, a): a for a in conversion_args}
            
            with tqdm(total=len(conversion_args), desc="Converting") as pbar:
                for future in as_completed(futures):
                    ok, msg = future.result()
                    if ok:
                        success += 1
                    else:
                        fail += 1
                        tqdm.write(msg)
                    pbar.update(1)
        
        print(f"\nConverted: {success} OK, {fail} failed")
    
    # Generate reference files
    print("\n" + "=" * 60)
    print("GENERATING VIEWER")
    print("=" * 60)
    
    create_metadata_csv(tile_metadata, output_path)
    
    overview_info = create_overview_map(tile_metadata, output_path)
    
    if overview_info:
        grid_info = build_geographic_grid(tile_metadata)
        create_html_viewer(tile_metadata, overview_info, grid_info, output_path, args.format)
    
    print("\n" + "=" * 60)
    print("COMPLETE")
    print("=" * 60)
    print(f"\nOpen {output_path / 'viewer.html'} in your browser")
    print("\nViewer features:")
    print("  • Click overview map to open any tile")
    print("  • Pan: drag with mouse")
    print("  • Zoom: mouse wheel or +/- keys")
    print("  • Navigate: arrow keys = N/S/E/W neighbors")
    print("  • Adjust: brightness, contrast, invert")
    print("  • Coordinates: shown on hover")
    print("\nLook for:")
    print("  • Linear features (roads, walls, ditches)")
    print("  • Circular/rectangular anomalies (mounds, foundations)")
    print("  • Regular patterns (field systems)")
    print("  • Subtle elevation changes")


if __name__ == '__main__':
    main()
