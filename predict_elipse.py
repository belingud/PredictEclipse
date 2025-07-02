#!/usr/bin/env python3
"""
Advanced Eclipse Prediction Script

A comprehensive tool for predicting solar and lunar eclipses using JPL ephemeris data.
Features automatic ephemeris selection, improved calculation accuracy, and robust error handling.

Author: Enhanced version with improved algorithms and documentation
License: MIT
"""

import argparse
import logging
import sys
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List
from urllib.parse import urljoin

import numpy as np
import requests
from bs4 import BeautifulSoup
from skyfield.api import load
from skyfield.almanac import find_discrete, moon_phases

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Constants for celestial mechanics calculations
MOON_RADIUS_KM = 1737.4  # Mean radius of the Moon in kilometers
SUN_RADIUS_KM = 695700  # Mean radius of the Sun in kilometers
EARTH_RADIUS_KM = 6371  # Mean radius of the Earth in kilometers

# Eclipse detection thresholds (in degrees)
ECLIPSE_ANGLE_THRESHOLD = 0.6  # Maximum angle for eclipse possibility
NODE_DISTANCE_TOTAL = 0.5  # Node distance threshold for total eclipses
NODE_DISTANCE_PARTIAL = 1.2  # Node distance threshold for partial eclipses
NODE_DISTANCE_PENUMBRAL = 1.5  # Node distance threshold for penumbral eclipses

# Comprehensive ephemeris metadata
EPHEMERIS_METADATA = {
    # DE400 series - Legacy but still useful
    "de405.bsp": {
        "start_year": 1600,
        "end_year": 2200,
        "size_mb": 63,
        "description": "Legacy standard ephemeris (1997-2003), reliable for historical work",
        "accuracy": "high",
        "priority": 3,
    },
    "de406.bsp": {
        "start_year": -3000,
        "end_year": 3000,
        "size_mb": 190,
        "description": "Extended historical ephemeris covering 6000 years",
        "accuracy": "high",
        "priority": 4,
    },
    # DE410-420 series - Improved accuracy
    "de410.bsp": {
        "start_year": 1900,
        "end_year": 2100,
        "size_mb": 65,
        "description": "Mars exploration era ephemeris with planetary improvements",
        "accuracy": "high",
        "priority": 3,
    },
    "de414.bsp": {
        "start_year": 1900,
        "end_year": 2100,
        "size_mb": 58,
        "description": "Enhanced planet and moon positions (2005 release)",
        "accuracy": "high",
        "priority": 3,
    },
    "de418.bsp": {
        "start_year": 1900,
        "end_year": 2200,
        "size_mb": 60,
        "description": "Further refined ephemeris with improved accuracy",
        "accuracy": "high",
        "priority": 3,
    },
    "de421.bsp": {
        "start_year": 1900,
        "end_year": 2050,
        "size_mb": 17,
        "description": "Popular compact ephemeris, excellent for modern applications",
        "accuracy": "high",
        "priority": 5,
    },
    "de422.bsp": {
        "start_year": -3000,
        "end_year": 3000,
        "size_mb": 623,
        "description": "Long-range ephemeris with superior lunar orbit accuracy",
        "accuracy": "very_high",
        "priority": 4,
    },
    # DE430+ series - Modern high-precision ephemerides
    "de430.bsp": {
        "start_year": 1550,
        "end_year": 2650,
        "size_mb": 115,
        "description": "NASA mission standard, exceptional precision (2013+)",
        "accuracy": "very_high",
        "priority": 6,
    },
    "de430t.bsp": {
        "start_year": 1950,
        "end_year": 2050,
        "size_mb": 128,
        "description": "Truncated DE430 optimized for 20th-21st centuries",
        "accuracy": "very_high",
        "priority": 6,
    },
    "de431.bsp": {
        "start_year": -13000,
        "end_year": 17000,
        "size_mb": 3400,
        "description": "Ultra-long span covering 30,000 years of solar system evolution",
        "accuracy": "very_high",
        "priority": 2,
    },
    "de432.bsp": {
        "start_year": 1950,
        "end_year": 2050,
        "size_mb": 65,
        "description": "Mars-focused ephemeris with enhanced red planet accuracy",
        "accuracy": "very_high",
        "priority": 4,
    },
    "de436.bsp": {
        "start_year": 1850,
        "end_year": 2150,
        "size_mb": 18,
        "description": "New Horizons mission optimized, excellent for outer planets",
        "accuracy": "very_high",
        "priority": 5,
    },
    "de438.bsp": {
        "start_year": 1950,
        "end_year": 2050,
        "size_mb": 160,
        "description": "Enhanced asteroid mass models for improved accuracy",
        "accuracy": "very_high",
        "priority": 4,
    },
    # DE440+ series - Latest generation
    "de440.bsp": {
        "start_year": 1550,
        "end_year": 2650,
        "size_mb": 114,
        "description": "Current JPL standard with state-of-the-art precision (2020+)",
        "accuracy": "exceptional",
        "priority": 7,
    },
    "de440s.bsp": {
        "start_year": -13000,
        "end_year": 17000,
        "size_mb": 32,
        "description": "Long-term DE440 variant, slightly reduced precision for efficiency",
        "accuracy": "very_high",
        "priority": 5,
    },
    "de440t.bsp": {
        "start_year": 1850,
        "end_year": 2150,
        "size_mb": 22,
        "description": "Optimized DE440 for modern era, best balance of size and accuracy",
        "accuracy": "exceptional",
        "priority": 7,
    },
    "de441.bsp": {
        "start_year": -13200,
        "end_year": 17191,
        "size_mb": 770,
        "description": "Extended precision DE440 variant for research applications",
        "accuracy": "exceptional",
        "priority": 6,
    },
    # Specialized ephemerides
    "jup310.bsp": {
        "start_year": 1900,
        "end_year": 2100,
        "size_mb": 38,
        "description": "Specialized Jupiter system ephemeris (moons and rings)",
        "accuracy": "high",
        "priority": 2,
    },
    "sat360xl.bsp": {
        "start_year": 1950,
        "end_year": 2050,
        "size_mb": 34,
        "description": "Specialized Saturn system ephemeris (moons and rings)",
        "accuracy": "high",
        "priority": 2,
    },
    "mar097.bsp": {
        "start_year": 1950,
        "end_year": 2050,
        "size_mb": 12,
        "description": "Mars system ephemeris optimized for lander missions",
        "accuracy": "high",
        "priority": 2,
    },
}


class EphemerisManager:
    """Manages ephemeris file selection, loading, and fallback strategies."""

    def __init__(self):
        self.available_online = {}
        self.load_timeout = 30

    def fetch_available_files(self) -> Dict[str, Dict[str, str]]:
        """
        Fetch available ephemeris files from JPL NAIF repositories.

        Returns:
            Dictionary of available files with metadata
        """
        repositories = [
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
            "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/satellites/",
        ]

        available_files = {}

        for repo_url in repositories:
            try:
                logger.info(f"Scanning repository: {repo_url}")
                response = requests.get(repo_url, timeout=self.load_timeout)
                response.raise_for_status()

                soup = BeautifulSoup(response.text, "html.parser")

                # Parse directory listing for .bsp files
                for link in soup.find_all("a", href=re.compile(r"\.bsp$", re.IGNORECASE)):
                    filename = link.get("href")
                    if not filename:
                        continue

                    # Extract file metadata from table row
                    parent_row = link.find_parent("tr")
                    if parent_row:
                        cells = parent_row.find_all("td")
                        size_info = cells[2].text.strip() if len(cells) > 2 else "Unknown"
                        date_info = cells[1].text.strip() if len(cells) > 1 else "Unknown"

                        # Convert size to human-readable format
                        size_display = self._format_file_size(size_info)

                        available_files[filename] = {
                            "source_url": repo_url,
                            "download_url": urljoin(repo_url, filename),
                            "size": size_display,
                            "modified_date": date_info,
                        }

            except requests.RequestException as e:
                logger.warning(f"Failed to access repository {repo_url}: {e}")
            except Exception as e:
                logger.error(f"Unexpected error parsing {repo_url}: {e}")

        self.available_online = available_files
        return available_files

    def _format_file_size(self, size_str: str) -> str:
        """Convert file size string to human-readable format."""
        try:
            size_bytes = int(size_str)
            if size_bytes >= 1024**3:
                return f"{size_bytes / (1024**3):.1f} GB"
            elif size_bytes >= 1024**2:
                return f"{size_bytes / (1024**2):.1f} MB"
            elif size_bytes >= 1024:
                return f"{size_bytes / 1024:.1f} KB"
            else:
                return f"{size_bytes} bytes"
        except (ValueError, TypeError):
            return size_str

    def select_optimal_ephemeris(self, start_date: datetime, end_date: datetime) -> str:
        """
        Intelligently select the best ephemeris file for the given date range.

        Args:
            start_date: Beginning of prediction period
            end_date: End of prediction period

        Returns:
            Name of the optimal ephemeris file
        """
        start_year = start_date.year
        end_year = end_date.year
        prediction_span = end_year - start_year

        # Scoring algorithm for ephemeris selection
        candidates = []

        for filename, metadata in EPHEMERIS_METADATA.items():
            # Check if ephemeris covers the required time range
            if start_year < metadata["start_year"] or end_year > metadata["end_year"]:
                continue

            # Calculate various scoring factors
            coverage_span = metadata["end_year"] - metadata["start_year"]
            efficiency_score = 1.0 - (metadata["size_mb"] / 1000.0)  # Prefer smaller files
            coverage_score = min(
                1.0, prediction_span / coverage_span
            )  # Prefer appropriate coverage
            priority_score = metadata["priority"] / 10.0  # Use priority rating

            # Accuracy bonus
            accuracy_bonus = {"exceptional": 0.3, "very_high": 0.2, "high": 0.1}.get(
                metadata["accuracy"], 0.0
            )

            # Penalize specialized ephemerides for general use
            general_use_bonus = 0.1 if filename.startswith("de") else -0.2

            # Calculate final score
            total_score = (
                priority_score * 0.4
                + efficiency_score * 0.2
                + coverage_score * 0.2
                + accuracy_bonus
                + general_use_bonus
            )

            candidates.append((filename, total_score, metadata))

        if not candidates:
            # Fallback selection for edge cases
            logger.warning(f"No ephemeris fully covers {start_year}-{end_year}")
            fallback_options = ["de440t.bsp", "de430t.bsp", "de421.bsp"]
            for fallback in fallback_options:
                if fallback in EPHEMERIS_METADATA:
                    logger.info(f"Using fallback ephemeris: {fallback}")
                    return fallback
            return "de421.bsp"  # Ultimate fallback

        # Select the highest-scoring candidate
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_file, score, metadata = candidates[0]

        logger.info(f"Selected ephemeris: {best_file} (score: {score:.3f})")
        logger.info(f"  Coverage: {metadata['start_year']}-{metadata['end_year']}")
        logger.info(f"  Size: ~{metadata['size_mb']}MB")
        logger.info(f"  Description: {metadata['description']}")

        return best_file

    def load_with_fallback(self, ephemeris_file: str, start_date: datetime, end_date: datetime):
        """
        Load ephemeris with intelligent fallback strategy.

        Args:
            ephemeris_file: Primary ephemeris file to load
            start_date: Start of prediction period
            end_date: End of prediction period

        Returns:
            Loaded ephemeris object
        """
        # Priority-ordered fallback sequence
        fallback_sequence = [
            ephemeris_file,
            "de440t.bsp",  # Modern, efficient
            "de430t.bsp",  # Reliable alternative
            "de421.bsp",  # Compact, widely available
            "de440s.bsp",  # Long-term coverage
        ]

        # Remove duplicates while preserving order
        unique_sequence = []
        seen = set()
        for eph in fallback_sequence:
            if eph not in seen:
                unique_sequence.append(eph)
                seen.add(eph)

        last_error = None
        for i, filename in enumerate(unique_sequence):
            try:
                logger.info(f"Loading ephemeris: {filename}")
                ephemeris = load(filename)

                # Validate time range coverage
                if filename in EPHEMERIS_METADATA:
                    self._validate_time_coverage(filename, start_date, end_date)

                if i > 0:
                    logger.warning(f"Successfully loaded fallback ephemeris: {filename}")

                return ephemeris

            except Exception as e:
                last_error = e
                logger.warning(f"Failed to load {filename}: {e}")

                if i < len(unique_sequence) - 1:
                    logger.info("Attempting next fallback option...")

        # If all options failed
        raise RuntimeError(
            f"Unable to load any ephemeris file. Last error: {last_error}"
        ) from last_error

    def _validate_time_coverage(self, filename: str, start_date: datetime, end_date: datetime):
        """Validate that ephemeris covers the required time range."""
        metadata = EPHEMERIS_METADATA[filename]
        start_year, end_year = start_date.year, end_date.year

        if start_year < metadata["start_year"] or end_year > metadata["end_year"]:
            logger.warning(
                f"Time range {start_year}-{end_year} may exceed reliable coverage of "
                f"{filename} ({metadata['start_year']}-{metadata['end_year']})"
            )


class EclipseCalculator:
    """Advanced eclipse calculation engine with improved algorithms."""

    def __init__(self, ephemeris, timescale):
        self.eph = ephemeris
        self.ts = timescale
        self.earth = ephemeris["earth"]
        self.moon = ephemeris["moon"]
        self.sun = ephemeris["sun"]

    def calculate_apparent_diameter(self, radius_km: float, distance_km: float) -> float:
        """
        Calculate apparent diameter of a celestial body.

        Args:
            radius_km: Physical radius in kilometers
            distance_km: Distance to observer in kilometers

        Returns:
            Apparent diameter in arcminutes
        """
        angular_radius = np.arctan(radius_km / distance_km)
        return (
            np.degrees(angular_radius) * 60 * 2
        )  # Convert to arcminutes, multiply by 2 for diameter

    def calculate_separation_angle(self, pos1: np.ndarray, pos2: np.ndarray) -> float:
        """
        Calculate angular separation between two position vectors.

        Args:
            pos1: First position vector
            pos2: Second position vector

        Returns:
            Angular separation in degrees
        """
        dot_product = np.dot(pos1, pos2)
        mag1 = np.linalg.norm(pos1)
        mag2 = np.linalg.norm(pos2)

        # Prevent numerical errors
        cos_angle = np.clip(dot_product / (mag1 * mag2), -1.0, 1.0)
        return np.degrees(np.arccos(cos_angle))

    def calculate_node_distance(self, moon_position) -> float:
        """
        Calculate distance from lunar orbital node (improved method).

        Args:
            moon_position: Moon's ecliptic position

        Returns:
            Node distance in degrees
        """
        ecliptic_coords = moon_position.ecliptic_position().km

        # Calculate ecliptic latitude (distance from ecliptic plane)
        x, y, z = ecliptic_coords
        ecliptic_distance = np.sqrt(x**2 + y**2)

        if ecliptic_distance > 0:
            latitude = np.degrees(np.arctan2(abs(z), ecliptic_distance))
        else:
            latitude = 0.0

        return latitude

    def determine_eclipse_type(self, context: Dict) -> str:
        """
        Determine eclipse type based on geometric conditions.

        Args:
            context: Dictionary containing calculation context

        Returns:
            Eclipse type string
        """
        angle = context["angle"]
        node_distance = context["node_distance"]
        moon_diameter = context["moon_diameter"]
        sun_diameter = context["sun_diameter"]
        eclipse_class = context["eclipse_class"]

        if eclipse_class == "solar":
            if angle > ECLIPSE_ANGLE_THRESHOLD:
                return "None"
            elif node_distance < NODE_DISTANCE_TOTAL:
                if moon_diameter >= sun_diameter * 1.01:  # 1% margin for totality
                    return "Total"
                elif moon_diameter <= sun_diameter * 0.99:  # 1% margin for annularity
                    return "Annular"
                else:
                    return "Hybrid"  # Rare hybrid eclipses
            elif node_distance < NODE_DISTANCE_PARTIAL:
                return "Partial"
            else:
                return "None"

        else:  # lunar eclipse
            if abs(angle - 180) > 1.5:  # More lenient threshold for lunar eclipses
                return "None"
            elif node_distance < NODE_DISTANCE_TOTAL:
                return "Total"
            elif node_distance < NODE_DISTANCE_PARTIAL:
                return "Partial"
            elif node_distance < NODE_DISTANCE_PENUMBRAL:
                return "Penumbral"
            else:
                return "None"

    def predict_lunar_eclipses(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Predict lunar eclipses with enhanced accuracy.

        Args:
            start_date: Start of prediction period
            end_date: End of prediction period

        Returns:
            List of eclipse event dictionaries
        """
        logger.info("Calculating lunar eclipse predictions...")

        t0 = self.ts.utc(start_date.year, start_date.month, start_date.day)
        t1 = self.ts.utc(end_date.year, end_date.month, end_date.day)

        times, phases = find_discrete(t0, t1, moon_phases(self.eph))
        eclipses = []

        for time, phase in zip(times, phases, strict=True):
            if phase != 2:  # Skip non-full moons
                continue

            # Calculate positions and distances
            moon_pos = self.earth.at(time).observe(self.moon).apparent()
            sun_pos = self.earth.at(time).observe(self.sun).apparent()

            earth_moon_dist = np.linalg.norm(moon_pos.position.km)
            earth_sun_dist = np.linalg.norm(sun_pos.position.km)

            # Calculate apparent diameters
            moon_diameter = self.calculate_apparent_diameter(MOON_RADIUS_KM, earth_moon_dist)
            sun_diameter = self.calculate_apparent_diameter(SUN_RADIUS_KM, earth_sun_dist)

            # Calculate geometric parameters
            separation_angle = self.calculate_separation_angle(
                sun_pos.position.km, moon_pos.position.km
            )
            node_distance = self.calculate_node_distance(moon_pos)

            # Determine eclipse type
            context = {
                "angle": separation_angle,
                "node_distance": node_distance,
                "moon_diameter": moon_diameter,
                "sun_diameter": sun_diameter,
                "eclipse_class": "lunar",
            }

            eclipse_type = self.determine_eclipse_type(context)

            if eclipse_type != "None":
                # Convert to local time
                utc_time = datetime.strptime(time.utc_iso(), "%Y-%m-%dT%H:%M:%SZ")
                local_time = utc_time.replace(tzinfo=timezone.utc).astimezone()

                eclipse_info = {
                    "type": eclipse_type,
                    "time_utc": utc_time,
                    "time_local": local_time,
                    "separation_angle": separation_angle,
                    "node_distance": node_distance,
                    "earth_moon_distance": earth_moon_dist,
                    "earth_sun_distance": earth_sun_dist,
                    "moon_apparent_diameter": moon_diameter,
                    "sun_apparent_diameter": sun_diameter,
                }

                eclipses.append(eclipse_info)

        return eclipses

    def predict_solar_eclipses(self, start_date: datetime, end_date: datetime) -> List[Dict]:
        """
        Predict solar eclipses with enhanced accuracy.

        Args:
            start_date: Start of prediction period
            end_date: End of prediction period

        Returns:
            List of eclipse event dictionaries
        """
        logger.info("Calculating solar eclipse predictions...")

        t0 = self.ts.utc(start_date.year, start_date.month, start_date.day)
        t1 = self.ts.utc(end_date.year, end_date.month, end_date.day)

        times, phases = find_discrete(t0, t1, moon_phases(self.eph))
        eclipses = []

        for time, phase in zip(times, phases, strict=True):
            if phase != 0:  # Skip non-new moons
                continue

            # Calculate positions and distances
            moon_pos = self.earth.at(time).observe(self.moon).apparent()
            sun_pos = self.earth.at(time).observe(self.sun).apparent()

            earth_moon_dist = np.linalg.norm(moon_pos.position.km)
            earth_sun_dist = np.linalg.norm(sun_pos.position.km)

            # Calculate apparent diameters
            moon_diameter = self.calculate_apparent_diameter(MOON_RADIUS_KM, earth_moon_dist)
            sun_diameter = self.calculate_apparent_diameter(SUN_RADIUS_KM, earth_sun_dist)

            # Calculate geometric parameters
            separation_angle = self.calculate_separation_angle(
                sun_pos.position.km, moon_pos.position.km
            )
            node_distance = self.calculate_node_distance(moon_pos)

            # Determine eclipse type
            context = {
                "angle": separation_angle,
                "node_distance": node_distance,
                "moon_diameter": moon_diameter,
                "sun_diameter": sun_diameter,
                "eclipse_class": "solar",
            }

            eclipse_type = self.determine_eclipse_type(context)

            if eclipse_type != "None":
                # Convert to local time
                utc_time = datetime.strptime(time.utc_iso(), "%Y-%m-%dT%H:%M:%SZ")
                local_time = utc_time.replace(tzinfo=timezone.utc).astimezone()

                eclipse_info = {
                    "type": eclipse_type,
                    "time_utc": utc_time,
                    "time_local": local_time,
                    "separation_angle": separation_angle,
                    "node_distance": node_distance,
                    "earth_moon_distance": earth_moon_dist,
                    "earth_sun_distance": earth_sun_dist,
                    "moon_apparent_diameter": moon_diameter,
                    "sun_apparent_diameter": sun_diameter,
                    "size_ratio": moon_diameter / sun_diameter if sun_diameter > 0 else 0,
                }

                eclipses.append(eclipse_info)

        return eclipses


class EclipseReporter:
    """Handles formatting and display of eclipse prediction results."""

    @staticmethod
    def format_eclipse_report(eclipses: List[Dict], eclipse_type: str) -> None:
        """
        Generate a formatted report of eclipse predictions.

        Args:
            eclipses: List of eclipse event dictionaries
            eclipse_type: Type of eclipse ("solar" or "lunar")
        """
        if not eclipses:
            print(f"⚠️  No {eclipse_type} eclipses found in the specified time range.")
            return

        eclipse_icon = "🌑" if eclipse_type == "solar" else "🌕"
        eclipse_name = eclipse_type.capitalize()

        print(f"\n{eclipse_icon} {eclipse_name} Eclipse Predictions ({len(eclipses)} found)")
        print("=" * 80)

        for i, eclipse in enumerate(eclipses, 1):
            print(f"\n#{i:2d}. {eclipse['type']} {eclipse_name} Eclipse")
            print(f"     Date & Time: {eclipse['time_local'].strftime('%Y-%m-%d %H:%M:%S %Z')}")
            print(f"     UTC Time:    {eclipse['time_utc'].strftime('%Y-%m-%d %H:%M:%S UTC')}")

            if eclipse_type == "solar":
                print(f"     Separation:  {eclipse['separation_angle']:.3f}°")
            else:
                print(f"     Sun-Earth-Moon Angle: {eclipse['separation_angle']:.3f}°")

            print(f"     Node Distance: {eclipse['node_distance']:.2f}°")
            print(f"     Earth-Moon Distance: {eclipse['earth_moon_distance']:,.0f} km")
            print(f"     Earth-Sun Distance:  {eclipse['earth_sun_distance']:,.0f} km")
            print(f"     Moon Apparent Diameter: {eclipse['moon_apparent_diameter']:.2f}'")
            print(f"     Sun Apparent Diameter:  {eclipse['sun_apparent_diameter']:.2f}'")

            if eclipse_type == "solar" and "size_ratio" in eclipse:
                ratio = eclipse["size_ratio"]
                print(f"     Moon/Sun Size Ratio: {ratio:.3f}", end="")
                if eclipse["type"] == "Total":
                    print(" (>1.0 → Total eclipse possible)")
                elif eclipse["type"] == "Annular":
                    print(" (<1.0 → Annular eclipse)")
                elif eclipse["type"] == "Hybrid":
                    print(" (≈1.0 → Hybrid eclipse)")
                else:
                    print()

    @staticmethod
    def display_ephemeris_catalog() -> None:
        """Display comprehensive ephemeris catalog information."""
        print("\n📊 Comprehensive Ephemeris Catalog")
        print("=" * 100)

        categories = {
            "Modern High-Precision (Recommended)": [
                "de440.bsp",
                "de440t.bsp",
                "de440s.bsp",
                "de441.bsp",
            ],
            "NASA Mission Standard": [
                "de430.bsp",
                "de430t.bsp",
                "de432.bsp",
                "de436.bsp",
                "de438.bsp",
            ],
            "Popular General-Purpose": ["de421.bsp", "de422.bsp", "de418.bsp"],
            "Historical Coverage": ["de406.bsp", "de431.bsp", "de422.bsp"],
            "Specialized Systems": ["jup310.bsp", "sat360xl.bsp", "mar097.bsp"],
        }

        for category, files in categories.items():
            print(f"\n{category}:")
            print("-" * len(category))

            for filename in files:
                if filename in EPHEMERIS_METADATA:
                    meta = EPHEMERIS_METADATA[filename]
                    print(
                        f"  {filename:12} | {meta['start_year']:5d}-{meta['end_year']:5d} | "
                        f"{meta['size_mb']:4.0f}MB | {meta['accuracy']:12} | {meta['description']}"
                    )

        print(f"\nTotal ephemeris files in catalog: {len(EPHEMERIS_METADATA)}")
        print("\nAccuracy levels: exceptional > very_high > high")
        print("Priority scoring: Higher numbers indicate better general-purpose suitability")


def parse_arguments() -> argparse.Namespace:
    """Parse and validate command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced Eclipse Prediction Tool",
        epilog="""
Examples:
  %(prog)s --type solar --start 2024-01-01 --end 2026-12-31
  %(prog)s --type lunar --start 2025-06-01 --end 2025-12-31 --ephemeris de440t.bsp
  %(prog)s --catalog
  %(prog)s --online-check
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--type",
        "-t",
        choices=["solar", "lunar"],
        help="Type of eclipse to predict (solar or lunar)",
    )

    parser.add_argument(
        "--start", "-s", type=str, help="Start date in YYYY-MM-DD format (default: today)"
    )

    parser.add_argument(
        "--end", "-e", type=str, help="End date in YYYY-MM-DD format (default: 2 years from start)"
    )

    parser.add_argument(
        "--ephemeris",
        "--eph",
        type=str,
        help="Specific ephemeris file to use (default: auto-select optimal)",
    )

    parser.add_argument(
        "--catalog",
        "-c",
        action="store_true",
        help="Display comprehensive ephemeris catalog and exit",
    )

    parser.add_argument(
        "--online-check",
        "-o",
        action="store_true",
        help="Check online repositories for available ephemeris files",
    )

    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging output"
    )

    parser.add_argument("--output", "-f", type=str, help="Output results to file (CSV format)")

    return parser.parse_args()


def validate_date_range(start_date: datetime, end_date: datetime) -> None:
    """Validate the date range for predictions."""
    if start_date >= end_date:
        raise ValueError("Start date must be before end date")

    time_span = end_date - start_date
    if time_span.days > 365 * 20:  # 20 years
        logger.warning(
            f"Large time span ({time_span.days} days) may require significant computation time"
        )
        response = input("Continue with large time span? (y/N): ")
        if response.lower() != "y":
            sys.exit(0)


def save_results_to_csv(eclipses: List[Dict], filename: str, eclipse_type: str) -> None:
    """Save eclipse prediction results to CSV file."""
    try:
        import csv

        fieldnames = [
            "eclipse_type",
            "date_local",
            "time_local",
            "date_utc",
            "time_utc",
            "separation_angle_deg",
            "node_distance_deg",
            "earth_moon_distance_km",
            "earth_sun_distance_km",
            "moon_apparent_diameter_arcmin",
            "sun_apparent_diameter_arcmin",
        ]

        if eclipse_type == "solar":
            fieldnames.append("moon_sun_size_ratio")

        with open(filename, "w", newline="", encoding="utf-8") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for eclipse in eclipses:
                row = {
                    "eclipse_type": eclipse["type"],
                    "date_local": eclipse["time_local"].strftime("%Y-%m-%d"),
                    "time_local": eclipse["time_local"].strftime("%H:%M:%S %Z"),
                    "date_utc": eclipse["time_utc"].strftime("%Y-%m-%d"),
                    "time_utc": eclipse["time_utc"].strftime("%H:%M:%S UTC"),
                    "separation_angle_deg": f"{eclipse['separation_angle']:.3f}",
                    "node_distance_deg": f"{eclipse['node_distance']:.2f}",
                    "earth_moon_distance_km": f"{eclipse['earth_moon_distance']:.0f}",
                    "earth_sun_distance_km": f"{eclipse['earth_sun_distance']:.0f}",
                    "moon_apparent_diameter_arcmin": f"{eclipse['moon_apparent_diameter']:.2f}",
                    "sun_apparent_diameter_arcmin": f"{eclipse['sun_apparent_diameter']:.2f}",
                }

                if eclipse_type == "solar" and "size_ratio" in eclipse:
                    row["moon_sun_size_ratio"] = f"{eclipse['size_ratio']:.3f}"

                writer.writerow(row)

        logger.info(f"Results saved to {filename}")

    except Exception as e:
        logger.error(f"Failed to save results to CSV: {e}")


def main():
    """Main application entry point."""
    try:
        args = parse_arguments()

        # Configure logging level
        if args.verbose:
            logging.getLogger().setLevel(logging.DEBUG)

        # Initialize ephemeris manager
        eph_manager = EphemerisManager()

        # Handle catalog display
        if args.catalog:
            EclipseReporter.display_ephemeris_catalog()
            return

        # Handle online repository check
        if args.online_check:
            print("🌐 Checking online ephemeris repositories...")
            online_files = eph_manager.fetch_available_files()

            if online_files:
                print(f"\n📡 Found {len(online_files)} ephemeris files online:")
                print("-" * 80)
                for filename, info in sorted(online_files.items()):
                    print(f"  {filename:15} | {info['size']:>10} | {info['modified_date']}")
                    print(f"                    | {info['download_url']}")
            else:
                print("❌ No online ephemeris files found or network error occurred")
            return

        # Validate required arguments
        if not args.type:
            print("❌ Error: --type argument is required")
            print("Use --help for usage information")
            sys.exit(1)

        # Parse and validate dates
        if args.start:
            try:
                start_date = datetime.strptime(args.start, "%Y-%m-%d")
            except ValueError:
                print(f"❌ Error: Invalid start date format '{args.start}'. Use YYYY-MM-DD")
                sys.exit(1)
        else:
            start_date = datetime.now()

        if args.end:
            try:
                end_date = datetime.strptime(args.end, "%Y-%m-%d")
            except ValueError:
                print(f"❌ Error: Invalid end date format '{args.end}'. Use YYYY-MM-DD")
                sys.exit(1)
        else:
            end_date = start_date + timedelta(days=730)  # 2 years default

        # Validate date range
        validate_date_range(start_date, end_date)

        # Select ephemeris file
        if args.ephemeris:
            ephemeris_file = args.ephemeris
            logger.info(f"Using specified ephemeris: {ephemeris_file}")
        else:
            ephemeris_file = eph_manager.select_optimal_ephemeris(start_date, end_date)

        # Load ephemeris with fallback strategy
        print("📡 Loading ephemeris data...")
        ephemeris = eph_manager.load_with_fallback(ephemeris_file, start_date, end_date)

        # Initialize calculator and perform predictions
        calculator = EclipseCalculator(ephemeris, load.timescale())

        print(
            f"\n🔮 Predicting {args.type} eclipses from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
        )

        if args.type == "lunar":
            eclipses = calculator.predict_lunar_eclipses(start_date, end_date)
        else:
            eclipses = calculator.predict_solar_eclipses(start_date, end_date)

        # Display results
        EclipseReporter.format_eclipse_report(eclipses, args.type)

        # Save to file if requested
        if args.output and eclipses:
            save_results_to_csv(eclipses, args.output, args.type)

        # Summary statistics
        if eclipses:
            print(f"\n📈 Summary: Found {len(eclipses)} {args.type} eclipse(s)")
            eclipse_types = {}
            for eclipse in eclipses:
                eclipse_types[eclipse["type"]] = eclipse_types.get(eclipse["type"], 0) + 1

            for eclipse_type, count in sorted(eclipse_types.items()):
                print(f"   {eclipse_type}: {count}")

    except KeyboardInterrupt:
        print("\n⚠️ Operation cancelled by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Application error: {e}")
        if args.verbose if "args" in locals() else False:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
