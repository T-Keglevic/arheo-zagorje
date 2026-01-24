# LiDAR Archaeological Survey Viewer v0.99

## User Documentation

---

## Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [How It Works](#how-it-works)
6. [Using the Viewer](#using-the-viewer)
7. [Coordinate Systems](#coordinate-systems)
8. [Command Line Reference](#command-line-reference)
9. [Troubleshooting](#troubleshooting)

---

## Overview

The LiDAR Archaeological Survey Viewer is a Python-based tool that converts Digital Elevation Model (DEM) files into an interactive web-based viewer optimized for archaeological prospection. It processes GeoTIFF files containing LiDAR terrain data and generates:

- High-quality hillshade visualizations using multi-directional lighting
- A seamless, Google Maps-style pan-and-zoom interface
- Coordinate conversion between projected (HTRS96/TM) and geographic (WGS84) systems
- Satellite imagery overlay for terrain comparison
- Address and coordinate search functionality

The viewer is designed for archaeologists, researchers, and enthusiasts who want to identify potential archaeological features (earthworks, burial mounds, ancient roads, field systems, etc.) in LiDAR terrain data.

---

## System Requirements

### Software Dependencies

- **Python 3.8+**
- **Required Python packages:**
  - `numpy` - numerical processing
  - `Pillow` (PIL) - image processing
  - `rasterio` - GeoTIFF reading
  - `tqdm` - progress bars (optional but recommended)

### Hardware Recommendations

- **RAM:** 8GB minimum, 16GB+ recommended for large datasets
- **Storage:** SSD recommended; output size is approximately 10-30% of input data
- **Display:** 1920×1080 minimum resolution recommended

### Input Data

- GeoTIFF files (.tif) containing elevation data
- Supported coordinate systems:
  - EPSG:3765 (HTRS96/TM - Croatia)
  - EPSG:3794 (D96/TM - Slovenia)
  - EPSG:32633 (UTM Zone 33N)
  - EPSG:32634 (UTM Zone 34N)
  - Local coordinate systems (auto-detected based on coordinate values)

---

## Installation

### Step 1: Install Python Dependencies

```bash
pip install numpy pillow rasterio tqdm
```

### Step 2: Download the Script

Save `convert_dmr_to_images.py` to your working directory.

### Step 3: Verify Installation

```bash
python convert_dmr_to_images.py --help
```

---

## Quick Start

### Basic Usage

```bash
python convert_dmr_to_images.py -i /path/to/tif/files -o /path/to/output
```

### Reference-Only Mode (Recommended for Re-runs)

If you've already processed the images and just want to regenerate the viewer:

```bash
python convert_dmr_to_images.py -i /path/to/tif/files -o /path/to/output --reference-only
```

### View the Results

Open `viewer.html` in the output directory with any modern web browser.

---

## How It Works

### Processing Pipeline

The tool performs the following steps:

#### 1. Tile Discovery and Metadata Extraction

The script scans the input directory for GeoTIFF files and extracts:
- Geographic bounds (left, right, top, bottom)
- Coordinate Reference System (CRS)
- Pixel dimensions
- Center coordinates

#### 2. Hillshade Generation

For each tile, a multi-directional hillshade is computed:

```
Final Hillshade = weighted average of hillshades from multiple sun angles
```

Default sun angles: 315°, 270°, 225°, 360° (NW, W, SW, N)

The hillshade algorithm uses the Horn method to calculate slope and aspect from the elevation grid, then computes illumination based on sun position.

#### 3. Image Export

Processed tiles are saved as JPEG (default, 85% quality) or PNG images, preserving the original tile naming.

#### 4. Overview Map Generation

A low-resolution overview map is created showing all tiles, used for navigation in the viewer.

#### 5. HTML Viewer Generation

An interactive HTML file is generated containing:
- All tile metadata (bounds, positions)
- Coordinate transformation functions
- Pan/zoom interface
- Search functionality
- Satellite overlay system

### Output Structure

```
output_directory/
├── viewer.html          # Main interactive viewer
├── overview_map.png     # Navigation overview image
├── tiles.csv            # Tile metadata spreadsheet
├── tile_001.jpg         # Processed tile images
├── tile_002.jpg
├── ...
└── tile_NNN.jpg
```

---

## Using the Viewer

### Interface Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Sidebar                    │  Main Map Area                │
│  ─────────                  │                               │
│  🔍 Archaeological Survey   │   [Zoom: 25% 50% 100% 200%]  │
│                             │   [Coordinates display]       │
│  SEARCH TILES               │                               │
│  [____________]             │                               │
│                             │      ┌─────────────────┐      │
│  SEARCH ADDRESS             │      │                 │      │
│  [____________]             │      │   LiDAR Tiles   │      │
│                             │      │                 │      │
│  GO TO COORDINATES          │      │   (pan & zoom)  │      │
│  [X/Lon] [Y/Lat]            │      │                 │      │
│  [Go] [Paste]               │      └─────────────────┘      │
│                             │                               │
│  MAP OVERLAY                ├───────────────────────────────┤
│  Satellite [────────] 0%    │ Drag to pan │ Scroll to zoom │
│  ☐ Show place names         │ C copy WGS84 │ M copy HTRS96 │
│                             │ L copy tile  │ [?]           │
│  DISPLAY                    └───────────────────────────────┘
│  Brightness [────────]      
│  Contrast   [────────]      
│  [Invert] [Reset] [Labels]  
│                             
│  OVERVIEW                   
│  ┌─────────────────┐        
│  │ [overview map]  │        
│  │    [viewport]   │        
│  └─────────────────┘        
│                             
│  TILE INFO                  
│  [hover info appears here]  
└─────────────────────────────┘
```

### Navigation

| Action | Method |
|--------|--------|
| Pan | Click and drag on the map |
| Zoom | Scroll wheel (zooms toward cursor) |
| Zoom (buttons) | Click 25%, 50%, 100%, 200%, or Fit All |
| Jump to location | Click on the overview map |

### Keyboard Shortcuts

| Key | Action |
|-----|--------|
| **C** | Copy WGS84 coordinates (latitude, longitude) |
| **M** | Copy HTRS96/TM coordinates (Easting, Northing) |
| **L** | Copy current tile filename |
| **I** | Toggle invert colors |
| **+** / **=** | Zoom in |
| **-** | Zoom out |
| **0** | Fit all tiles in view |
| **Esc** | Close info modal |

### Search Features

#### Tile Search
- Type part of a tile name to filter
- Use `*` as wildcard (e.g., `DMR*103*`)
- Click a result or use arrow keys + Enter to navigate
- Dropdown shows tile name and WGS84 coordinates

#### Address Search
- Type any place name, street, or address
- Searches Croatia and Slovenia via Photon/Nominatim APIs
- Partial names work (e.g., "Zagr" finds "Zagreb")
- Click a result to navigate to that location
- Works even for locations outside your LiDAR coverage

#### Coordinate Search
- Enter coordinates in either format:
  - **WGS84:** `46.137, 15.778` (decimal degrees)
  - **HTRS96:** `444265, 5111092` (meters)
- The system auto-detects the format based on value magnitude
- Click "Go" or press Enter to navigate
- "Paste" button parses coordinates from clipboard

### Satellite Overlay

The satellite slider controls visibility of aerial imagery beneath your LiDAR data:

| Slider Position | Result |
|-----------------|--------|
| 0% | LiDAR only (full opacity) |
| 50% | LiDAR semi-transparent over satellite |
| 100% | Satellite only (LiDAR invisible) |

This allows you to:
- Verify LiDAR features against modern terrain
- Identify buildings, roads, and vegetation
- Correlate archaeological features with current land use

**Place Names Checkbox:** Overlays settlement names, roads, and geographic labels from OpenStreetMap on top of your view.

### Display Adjustments

| Control | Effect |
|---------|--------|
| **Brightness** | Lighten (>100%) or darken (<100%) the image |
| **Contrast** | Increase (>100%) or decrease (<100%) contrast |
| **Invert** | Swap black/white; useful for seeing subtle features |
| **Reset** | Return to default display settings |
| **Labels** | Toggle tile boundary labels on/off |

### Tile Information Panel

When hovering over a tile, the sidebar shows:
- **Tile name** (filename without extension)
- **Size** in pixels
- **Easting range** (projected X coordinates)
- **Northing range** (projected Y coordinates)  
- **Center** in WGS84 (clickable to copy)

---

## Coordinate Systems

### Understanding the Two Systems

The viewer displays coordinates in two formats simultaneously:

#### HTRS96/TM (EPSG:3765) - Projected Coordinates

**Example:** `E: 444,264.9  N: 5,111,092.4`

This is Croatia's official coordinate system, a Transverse Mercator projection.

| Parameter | Value |
|-----------|-------|
| Ellipsoid | GRS80 |
| Central Meridian | 16.5°E |
| False Easting | 500,000 m |
| False Northing | 0 m |
| Scale Factor | 0.9999 |

**Reading the coordinates:**
- **Easting (E):** Meters east/west of the central meridian (16.5°E), plus 500,000m offset
  - E < 500,000 → west of 16.5°E
  - E > 500,000 → east of 16.5°E
- **Northing (N):** Meters north of the equator

**Advantages:** 
- Distances in meters are intuitive
- Easy to calculate straight-line distances
- No negative numbers

#### WGS84 (EPSG:4326) - Geographic Coordinates

**Example:** `46.137375, 15.778623`

This is the global standard used by GPS, Google Maps, and most web mapping.

| Component | Meaning |
|-----------|---------|
| Latitude (first number) | Degrees north of the equator |
| Longitude (second number) | Degrees east of the Prime Meridian |

**Advantages:**
- Universal - works anywhere on Earth
- Directly usable in Google Maps, GPS devices
- Standard for sharing locations online

### Conversion Example

```
HTRS96: E: 444,264.9, N: 5,111,092.4
        ↓
Easting: 444,265 - 500,000 = -55,735m (west of 16.5°E)
At 46°N: 1° longitude ≈ 77.8 km
Longitude: 16.5° - (55.735 / 77.8) ≈ 15.78°E

Northing: 5,111,092m from equator
Using ellipsoid geometry → 46.137°N
        ↓
WGS84: 46.137°N, 15.778°E
```

### Info Button

Click the **?** button in the help strip to display a detailed explanation of both coordinate systems within the viewer.

---

## Command Line Reference

### Basic Syntax

```bash
python convert_dmr_to_images.py -i INPUT -o OUTPUT [options]
```

### Required Arguments

| Argument | Description |
|----------|-------------|
| `-i`, `--input` | Input directory containing GeoTIFF files |
| `-o`, `--output` | Output directory for processed files |

### Optional Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--format` | `jpg` | Output format: `jpg` or `png` |
| `--quality` | `85` | JPEG quality (1-100) |
| `--reference-only` | off | Skip image processing; regenerate viewer only |
| `--sun-elevation` | `45` | Sun elevation angle in degrees |
| `--sun-azimuths` | `315,270,225,360` | Comma-separated sun azimuth angles |
| `--workers` | CPU count | Number of parallel processing workers |

### Examples

**High-quality PNG output:**
```bash
python convert_dmr_to_images.py -i ./dem_tiles -o ./output --format png
```

**Custom sun angles for enhanced shadow detail:**
```bash
python convert_dmr_to_images.py -i ./dem_tiles -o ./output --sun-azimuths 315,45,135,225
```

**Regenerate viewer after code update:**
```bash
python convert_dmr_to_images.py -i ./dem_tiles -o ./output --reference-only
```

---

## Troubleshooting

### Common Issues

#### "No valid tiles found"

**Cause:** The input directory contains no readable GeoTIFF files.

**Solutions:**
- Verify files have `.tif` extension
- Check files are valid GeoTIFFs with `gdalinfo filename.tif`
- Ensure read permissions on the files

#### Coordinates appear offset from Google Maps

**Cause:** Incorrect coordinate system detection.

**Solutions:**
- Check the console output during generation for "Detected CRS" and "Using projection"
- If auto-detection fails, verify your data's actual CRS
- Croatian data should use HTRS96 (central meridian 16.5°E)
- Slovenian data should use D96TM (central meridian 15.0°E)

#### Satellite layer doesn't align with LiDAR

**Cause:** Coordinate transformation mismatch.

**Solutions:**
- Verify the projection parameters match your data
- Check that WGS84 coordinates in the viewer match expected locations
- Compare a known landmark in both LiDAR and satellite views

#### Browser shows blank page or errors

**Cause:** Browser security restrictions on local files.

**Solutions:**
- Use a modern browser (Chrome, Firefox, Edge)
- Serve files through a local web server:
  ```bash
  cd output_directory
  python -m http.server 8000
  ```
  Then open `http://localhost:8000/viewer.html`

#### Address search returns no results

**Cause:** Network connectivity or API limitations.

**Solutions:**
- Verify internet connection
- Try more specific search terms
- Check browser console for CORS errors
- The Photon API may have rate limits; wait and retry

### Performance Tips

1. **Use JPEG format** for faster loading (smaller files than PNG)
2. **Process subsets** for testing before running full datasets
3. **Use `--reference-only`** when only viewer changes are needed
4. **SSD storage** dramatically improves tile loading speed
5. **Reduce browser zoom** (Ctrl+0) if scrolling feels slow with many tiles

---

## Credits and Data Sources

### Satellite Imagery
- ESRI World Imagery (ArcGIS)

### Map Labels
- CartoDB/CARTO labels layer
- OpenStreetMap contributors

### Address Search
- Photon by Komoot (primary)
- OpenStreetMap Nominatim (fallback)

### LiDAR Data
This tool is designed for use with LiDAR data from national mapping agencies such as:
- DGU - Državna geodetska uprava (Croatia)
- GURS - Geodetska uprava Republike Slovenije (Slovenia)

---

## Version History

### v0.99 (Current)
- Continuous scrolling map interface (Google Maps style)
- Dual coordinate display (HTRS96 + WGS84)
- Satellite imagery overlay with transparency control
- Place name labels overlay
- Address search with autocomplete
- Tile search with dropdown
- Coordinate search (both formats)
- Keyboard shortcuts for copying coordinates
- Coordinate systems info modal
- Auto-detection of Croatian/Slovenian projections

---

## License

This tool is provided for archaeological research and educational purposes.

LiDAR data may be subject to licensing terms from the originating national mapping agency. Users are responsible for compliance with applicable data usage terms.
