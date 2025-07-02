#!/usr/bin/env python3
"""
Test Suite: Solar Eclipse and Lunar Eclipse Prediction Script

Contains unit and integration tests for each component of predict_elipse.py.
"""

import unittest
from unittest import mock
import os
from datetime import datetime, timedelta
import tempfile
import numpy as np
from io import StringIO

# Import the module under test
from predict_elipse import (
    EphemerisManager,
    EclipseCalculator,
    EclipseReporter,
    validate_date_range,
    save_results_to_csv,
    main,
)


class TestEphemerisManager(unittest.TestCase):
    """Test the EphemerisManager class"""

    def setUp(self):
        """Set up the test environment"""
        self.temp_dir = tempfile.TemporaryDirectory()
        self.manager = EphemerisManager(self.temp_dir.name)

    def tearDown(self):
        """Clean up the test environment"""
        self.temp_dir.cleanup()

    def test_format_file_size(self):
        """Test file size formatting"""
        tests = [
            ("1024", "1.0 KB"),
            ("1048576", "1.0 MB"),
            ("1073741824", "1.0 GB"),
            ("500", "500 bytes"),
            ("abcd", "abcd"),  # Non-numeric input
            ("", ""),  # Empty input
        ]

        for input_str, expected in tests:
            with self.subTest(input=input_str):
                result = self.manager._format_file_size(input_str)
                self.assertEqual(result, expected)

    @mock.patch("requests.get")
    def test_fetch_available_files(self, mock_get):
        """Test fetching available ephemeris files"""
        # Mock the request response
        mock_response = mock.Mock()
        mock_response.text = """
        <html>
            <body>
                <table>
                    <tr>
                        <td><a href="de440.bsp">de440.bsp</a></td>
                        <td>2021-01-01</td>
                        <td>114000000</td>
                    </tr>
                </table>
            </body>
        </html>
        """
        mock_response.raise_for_status = mock.Mock()
        mock_get.return_value = mock_response

        result = self.manager.fetch_available_files()

        self.assertIn("de440.bsp", result)
        self.assertEqual(result["de440.bsp"]["size"], "108.7 MB")

    def test_select_optimal_ephemeris(self):
        """Test selecting the optimal ephemeris file"""
        # Test normal modern date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2025, 1, 1)
        ephemeris = self.manager.select_optimal_ephemeris(start_date, end_date)
        self.assertIn(ephemeris, ["de440t.bsp", "de440.bsp"])

        # Test long historical range
        start_date = datetime(1700, 1, 1)
        end_date = datetime(1800, 1, 1)
        ephemeris = self.manager.select_optimal_ephemeris(start_date, end_date)
        self.assertIn(ephemeris, ["de431.bsp", "de422.bsp", "de406.bsp", "de440.bsp"])

    @mock.patch("skyfield.iokit.download")
    @mock.patch("skyfield.api.Loader")
    def test_load_with_fallback(self, mock_loader, mock_download):
        """Test loading ephemeris and handling fallback strategy"""
        # Mock successful loading
        mock_ephemeris = mock.Mock()

        # Create special manager with mock loader
        special_manager = EphemerisManager(self.temp_dir.name)

        # Setup loader to succeed on first call, then fail, then succeed again (testing fallback)
        special_manager.loader = mock.Mock(
            side_effect=[
                mock_ephemeris,  # First call succeeds
                Exception("Failed to load primary"),  # Second call fails
                mock_ephemeris,  # Third call (fallback) succeeds
            ]
        )

        # Replace self.manager with our special version
        original_manager = self.manager
        self.manager = special_manager

        # Mock download function to do nothing
        mock_download.return_value = None

        try:
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 1, 1)

            # First call should succeed
            result = self.manager.load_with_fallback("de440t.bsp", start_date, end_date)
            self.assertEqual(result, mock_ephemeris)

            # Second call should fail on primary but succeed with fallback
            result = self.manager.load_with_fallback("de440t.bsp", start_date, end_date)
            self.assertEqual(result, mock_ephemeris)

        finally:
            # Restore original manager
            self.manager = original_manager

    @mock.patch("skyfield.iokit.download")
    @mock.patch("skyfield.api.Loader")
    def test_ephemeris_download(self, mock_loader, mock_download):
        """Test ephemeris download functionality"""
        # Mock the ephemeris file and loader
        mock_ephemeris = mock.Mock()

        # Create a special mock for the manager to bypass the actual loading
        special_manager = EphemerisManager(self.temp_dir.name)
        special_manager.loader = mock.Mock(return_value=mock_ephemeris)

        # Replace self.manager with our special version
        original_manager = self.manager
        self.manager = special_manager

        # Mock download function to do nothing
        mock_download.return_value = None

        try:
            # Test with a URL that would trigger download
            test_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440t.bsp"
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 1, 1)

            # Call load_with_fallback
            result = self.manager.load_with_fallback(test_url, start_date, end_date)

            # Verify results
            self.assertEqual(result, mock_ephemeris)
            self.manager.loader.assert_called_with(test_url)

        finally:
            # Restore original manager
            self.manager = original_manager

    @mock.patch("skyfield.iokit.download")
    @mock.patch("skyfield.api.Loader")
    def test_ephemeris_download_error(self, mock_loader, mock_download):
        """Test error handling during ephemeris download"""
        # Create a special mock for the manager to bypass the actual loading
        special_manager = EphemerisManager(self.temp_dir.name)

        # Set up the loader to raise exceptions for all ephemeris files
        special_manager.loader = mock.Mock(side_effect=ValueError("Failed to load ephemeris"))

        # Replace self.manager with our special version
        original_manager = self.manager
        self.manager = special_manager

        # Mock download function to do nothing
        mock_download.return_value = None

        try:
            # Test with a URL
            test_url = "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440t.bsp"
            start_date = datetime(2023, 1, 1)
            end_date = datetime(2025, 1, 1)

            # Should raise RuntimeError when all options fail
            with self.assertRaises(RuntimeError):
                self.manager.load_with_fallback(test_url, start_date, end_date)

            # Verify that loader was called for each fallback option
            self.assertEqual(special_manager.loader.call_count, 5)  # URL + 4 fallback options

        finally:
            # Restore original manager
            self.manager = original_manager


class TestEclipseCalculator(unittest.TestCase):
    """Test the EclipseCalculator class"""

    def setUp(self):
        """Set up the test environment"""
        # Create mock objects
        self.mock_ephemeris = {"earth": mock.Mock(), "moon": mock.Mock(), "sun": mock.Mock()}
        self.mock_ts = mock.Mock()
        self.calculator = EclipseCalculator(self.mock_ephemeris, self.mock_ts)

    def test_calculate_apparent_diameter(self):
        """Test calculating apparent diameter"""
        radius = 1737.4  # Moon radius (km)
        distance = 384400  # Average Earth-Moon distance (km)

        # Manually calculate expected result (angle radius = arctan(r/d) * 2 * 60 (convert to arc minutes))
        expected = np.degrees(np.arctan(radius / distance)) * 60 * 2

        result = self.calculator.calculate_apparent_diameter(radius, distance)
        self.assertAlmostEqual(result, expected, places=5)

    def test_calculate_separation_angle(self):
        """Test calculating separation angle"""
        # Vertical vector, expected 90 degrees
        pos1 = np.array([1, 0, 0])
        pos2 = np.array([0, 1, 0])
        result = self.calculator.calculate_separation_angle(pos1, pos2)
        self.assertAlmostEqual(result, 90.0, places=5)

        # Same direction, expected 0 degrees
        pos1 = np.array([1, 0, 0])
        pos2 = np.array([2, 0, 0])
        result = self.calculator.calculate_separation_angle(pos1, pos2)
        self.assertAlmostEqual(result, 0.0, places=5)

        # Opposite direction, expected 180 degrees
        pos1 = np.array([1, 0, 0])
        pos2 = np.array([-1, 0, 0])
        result = self.calculator.calculate_separation_angle(pos1, pos2)
        self.assertAlmostEqual(result, 180.0, places=5)

    def test_determine_eclipse_type(self):
        """Test determining eclipse type"""
        # Test solar eclipse type
        context = {
            "angle": 0.1,
            "node_distance": 0.1,
            "moon_diameter": 31.0,
            "sun_diameter": 30.0,
            "eclipse_class": "solar",
        }
        result = self.calculator.determine_eclipse_type(context)
        self.assertEqual(result, "Total")

        # Test lunar eclipse type
        context = {
            "angle": 179.0,
            "node_distance": 0.1,
            "moon_diameter": 31.0,
            "sun_diameter": 30.0,
            "eclipse_class": "lunar",
        }
        result = self.calculator.determine_eclipse_type(context)
        self.assertEqual(result, "Total")

        # Test no eclipse
        context = {
            "angle": 1.0,
            "node_distance": 2.0,
            "moon_diameter": 31.0,
            "sun_diameter": 30.0,
            "eclipse_class": "solar",
        }
        result = self.calculator.determine_eclipse_type(context)
        self.assertEqual(result, "None")

    @mock.patch("predict_elipse.find_discrete")
    @mock.patch("predict_elipse.moon_phases")
    def test_predict_lunar_eclipses(self, mock_moon_phases, mock_find_discrete):
        """Test lunar eclipse prediction"""
        # Mock full moon time
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # Mock find_discrete return value
        mock_time = mock.Mock()
        mock_time.utc_iso.return_value = "2023-05-05T12:00:00Z"
        mock_times = [mock_time]
        mock_phases = [2]  # Full moon phase
        mock_find_discrete.return_value = (mock_times, mock_phases)

        # Mock Earth, Moon, and Sun position observations
        mock_moon_obs = mock.Mock()
        mock_sun_obs = mock.Mock()

        # Set position vectors and apparent vectors
        mock_moon_obs.position.km = np.array([384400, 0, 0])
        mock_sun_obs.position.km = np.array([-150000000, 0, 0])

        # Set apparent method
        mock_moon_obs.apparent.return_value = mock_moon_obs
        mock_sun_obs.apparent.return_value = mock_sun_obs

        # Set ecliptic_position method
        mock_moon_obs.ecliptic_position.return_value = mock.Mock(km=np.array([384400, 0, 100]))

        # Mock Earth observation
        self.mock_ephemeris["earth"].at.return_value.observe.side_effect = [
            mock_moon_obs,  # First call returns Moon observation
            mock_sun_obs,  # Second call returns Sun observation
        ]

        # Execute prediction
        eclipses = self.calculator.predict_lunar_eclipses(start_date, end_date)

        # Verify results
        self.assertEqual(len(eclipses), 1)
        self.assertIn("type", eclipses[0])
        self.assertIn("time_utc", eclipses[0])
        self.assertIn("time_local", eclipses[0])

    @mock.patch("predict_elipse.find_discrete")
    @mock.patch("predict_elipse.moon_phases")
    def test_predict_solar_eclipses(self, mock_moon_phases, mock_find_discrete):
        """Test solar eclipse prediction"""
        # Similar to lunar eclipse prediction mock setup
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)

        # Mock new moon time
        mock_time = mock.Mock()
        mock_time.utc_iso.return_value = "2023-04-20T04:00:00Z"
        mock_times = [mock_time]
        mock_phases = [0]  # crescent moon phase (astronomy)
        mock_find_discrete.return_value = (mock_times, mock_phases)

        # Mock Earth, Moon, and Sun position observations (similar to lunar eclipse test)
        mock_moon_obs = mock.Mock()
        mock_sun_obs = mock.Mock()

        mock_moon_obs.position.km = np.array([384400, 0, 0])
        mock_sun_obs.position.km = np.array([150000000, 0, 0])

        mock_moon_obs.apparent.return_value = mock_moon_obs
        mock_sun_obs.apparent.return_value = mock_sun_obs

        mock_moon_obs.ecliptic_position.return_value = mock.Mock(km=np.array([384400, 0, 10]))

        self.mock_ephemeris["earth"].at.return_value.observe.side_effect = [
            mock_moon_obs,
            mock_sun_obs,
        ]

        # Execute prediction
        eclipses = self.calculator.predict_solar_eclipses(start_date, end_date)

        # Verify results
        self.assertEqual(len(eclipses), 1)
        self.assertIn("type", eclipses[0])
        self.assertIn("time_utc", eclipses[0])
        self.assertIn("time_local", eclipses[0])
        self.assertIn("size_ratio", eclipses[0])


class TestEclipseReporter(unittest.TestCase):
    """Test the EclipseReporter class"""

    def setUp(self):
        """Set up the test environment"""
        self.maxDiff = None

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_format_eclipse_report_solar(self, mock_stdout):
        """Test solar eclipse report formatting"""
        # Create example eclipse data
        utc_time = datetime(2023, 4, 20, 4, 0, 0)
        local_time = utc_time

        eclipses = [
            {
                "type": "Total",
                "time_utc": utc_time,
                "time_local": local_time,
                "separation_angle": 0.15,
                "node_distance": 0.1,
                "earth_moon_distance": 384400,
                "earth_sun_distance": 150000000,
                "moon_apparent_diameter": 31.0,
                "sun_apparent_diameter": 30.0,
                "size_ratio": 1.03,
            }
        ]

        EclipseReporter.format_eclipse_report(eclipses, "solar")

        output = mock_stdout.getvalue()
        self.assertIn("🌑 Solar Eclipse Predictions", output)
        self.assertIn("Total Solar Eclipse", output)
        self.assertIn("Moon/Sun Size Ratio: 1.030", output)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_format_eclipse_report_lunar(self, mock_stdout):
        """Test lunar eclipse report formatting"""
        # Create example lunar eclipse data
        utc_time = datetime(2023, 5, 5, 12, 0, 0)
        local_time = utc_time

        eclipses = [
            {
                "type": "Total",
                "time_utc": utc_time,
                "time_local": local_time,
                "separation_angle": 179.5,
                "node_distance": 0.1,
                "earth_moon_distance": 384400,
                "earth_sun_distance": 150000000,
                "moon_apparent_diameter": 31.0,
                "sun_apparent_diameter": 30.0,
            }
        ]

        EclipseReporter.format_eclipse_report(eclipses, "lunar")

        output = mock_stdout.getvalue()
        self.assertIn("🌕 Lunar Eclipse Predictions", output)
        self.assertIn("Total Lunar Eclipse", output)
        self.assertIn("Sun-Earth-Moon Angle: 179.500°", output)

    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_display_ephemeris_catalog(self, mock_stdout):
        """Test displaying ephemeris catalog"""
        EclipseReporter.display_ephemeris_catalog()

        output = mock_stdout.getvalue()
        self.assertIn("📊 Comprehensive Ephemeris Catalog", output)
        self.assertIn("Modern High-Precision (Recommended)", output)
        self.assertIn("de440.bsp", output)
        self.assertIn("de430.bsp", output)


class TestHelperFunctions(unittest.TestCase):
    """Test helper functions"""

    @mock.patch("sys.exit")
    @mock.patch("builtins.input", return_value="n")
    def test_validate_date_range(self, mock_input, mock_exit):
        """Test date range validation"""
        # Test valid date range
        start_date = datetime(2023, 1, 1)
        end_date = datetime(2023, 12, 31)
        validate_date_range(start_date, end_date)

        # Test invalid date range
        with self.assertRaises(ValueError):
            validate_date_range(end_date, start_date)

        # Test long date range
        long_end_date = start_date + timedelta(days=365 * 21)
        validate_date_range(start_date, long_end_date)
        mock_exit.assert_called_once()

    def test_save_results_to_csv(self):
        """Test saving results to CSV file"""
        # Create temporary file
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
            # Prepare test data
            utc_time = datetime(2023, 4, 20, 4, 0, 0)
            local_time = utc_time

            eclipses = [
                {
                    "type": "Total",
                    "time_utc": utc_time,
                    "time_local": local_time,
                    "separation_angle": 0.15,
                    "node_distance": 0.1,
                    "earth_moon_distance": 384400,
                    "earth_sun_distance": 150000000,
                    "moon_apparent_diameter": 31.0,
                    "sun_apparent_diameter": 30.0,
                    "size_ratio": 1.03,
                }
            ]

            # Save results
            save_results_to_csv(eclipses, tmp.name, "solar")

            # Verify CSV content
            with open(tmp.name, "r") as f:
                content = f.read()
                self.assertIn("eclipse_type,date_local", content)
                self.assertIn("Total,2023-04-20", content)

            # Clean up
            os.unlink(tmp.name)


class TestIntegration(unittest.TestCase):
    """Integration test"""

    @mock.patch("sys.argv", ["predict_elipse.py", "--catalog"])
    @mock.patch("sys.stdout", new_callable=StringIO)
    def test_main_catalog(self, mock_stdout):
        """Test main program displaying catalog"""
        main()
        output = mock_stdout.getvalue()
        self.assertIn("📊 Comprehensive Ephemeris Catalog", output)

    @mock.patch("skyfield.iokit.download")
    @mock.patch("predict_elipse.load.timescale")
    @mock.patch(
        "sys.argv",
        ["predict_elipse.py", "--type", "solar", "--start", "2023-01-01", "--end", "2023-01-10"],
    )
    @mock.patch("sys.stdout", new_callable=StringIO)
    @mock.patch("builtins.open", mock.mock_open(read_data=b"mock_ephemeris_data"))  # Binary mode
    def test_main_predict(self, mock_stdout, mock_ts, mock_download):
        """Test main program predicting"""
        # Set up mock objects for EphemerisManager
        with mock.patch("predict_elipse.EphemerisManager") as mock_manager_class:
            mock_manager_instance = mock.Mock()
            mock_manager_class.return_value = mock_manager_instance
            mock_manager_instance.select_optimal_ephemeris.return_value = "de440t.bsp"

            # Mock download function to do nothing
            mock_download.return_value = None

            # Mock ephemeris object
            mock_ephemeris = {"earth": mock.Mock(), "moon": mock.Mock(), "sun": mock.Mock()}
            mock_manager_instance.load_with_fallback.return_value = mock_ephemeris

            # Mock timescale object
            mock_ts_instance = mock.Mock()
            mock_ts.return_value = mock_ts_instance

            # Mock empty prediction results
            with mock.patch("predict_elipse.EclipseCalculator") as mock_calculator:
                mock_calculator_instance = mock.Mock()
                mock_calculator.return_value = mock_calculator_instance
                mock_calculator_instance.predict_solar_eclipses.return_value = []

                # Mock any file system checks
                with mock.patch("os.path.exists", return_value=True):
                    with mock.patch("os.makedirs"):
                        main()

                        # Verify output
                        output = mock_stdout.getvalue()
                        self.assertIn("🔮 Predicting solar eclipses", output)
                        self.assertIn("No solar eclipses found", output)

                        # Verify that load_with_fallback was called
                        mock_manager_instance.load_with_fallback.assert_called_once()

    @mock.patch("skyfield.iokit.download")
    @mock.patch("predict_elipse.load.timescale")
    @mock.patch(
        "sys.argv",
        ["predict_elipse.py", "--type", "solar", "--data-dir", "~/custom_data_dir"],
    )
    @mock.patch("sys.stdout", new_callable=StringIO)
    @mock.patch("os.path.expanduser")
    @mock.patch("os.makedirs", autospec=True)
    @mock.patch("builtins.open", mock.mock_open(read_data=b"mock_ephemeris_data"))  # Binary mode
    def test_main_with_custom_data_dir(
        self, mock_makedirs, mock_expanduser, mock_download, mock_stdout, mock_ts
    ):
        """Test main program with custom data directory"""
        # Mock path expansion
        mock_expanduser.return_value = "/home/user/custom_data_dir"

        # Set up mock objects
        with mock.patch("predict_elipse.EphemerisManager") as mock_manager:
            mock_manager_instance = mock.Mock()
            mock_manager.return_value = mock_manager_instance
            mock_manager_instance.select_optimal_ephemeris.return_value = "de440t.bsp"

            # Mock download function to do nothing
            mock_download.return_value = None

            # Mock ephemeris and time objects
            mock_ephemeris = {"earth": mock.Mock(), "moon": mock.Mock(), "sun": mock.Mock()}
            mock_manager_instance.load_with_fallback.return_value = mock_ephemeris
            mock_ts_instance = mock.Mock()
            mock_ts.return_value = mock_ts_instance

            # Mock empty prediction results
            with mock.patch("predict_elipse.EclipseCalculator") as mock_calculator:
                mock_calculator_instance = mock.Mock()
                mock_calculator.return_value = mock_calculator_instance
                mock_calculator_instance.predict_solar_eclipses.return_value = []

                with mock.patch("os.path.exists", return_value=True):
                    # Execute main with custom data directory
                    main()

                    # Verify EphemerisManager was created with correct path
                    mock_manager.assert_called_with("/home/user/custom_data_dir")

                    # Verify directory was created
                    mock_makedirs.assert_called_with("/home/user/custom_data_dir", exist_ok=True)

    @mock.patch("sys.argv", ["predict_elipse.py", "--online-check"])
    @mock.patch("sys.stdout", new_callable=StringIO)
    @mock.patch("predict_elipse.EphemerisManager")
    def test_main_online_check(self, mock_manager, mock_stdout):
        """Test the online repository check option"""
        # Mock the manager and its fetch_available_files method
        mock_manager_instance = mock.Mock()
        mock_manager.return_value = mock_manager_instance

        # Create sample return data
        online_files = {
            "de440.bsp": {
                "source_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
                "download_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de440.bsp",
                "size": "114 MB",
                "modified_date": "2021-01-01",
            },
            "de430.bsp": {
                "source_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/",
                "download_url": "https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de430.bsp",
                "size": "115 MB",
                "modified_date": "2013-07-31",
            },
        }

        mock_manager_instance.fetch_available_files.return_value = online_files

        # Execute main with --online-check
        main()

        # Verify output
        output = mock_stdout.getvalue()
        self.assertIn("🌐 Checking online ephemeris repositories", output)
        self.assertIn("Found 2 ephemeris files online", output)
        self.assertIn("de440.bsp", output)

        # Verify method was called
        mock_manager_instance.fetch_available_files.assert_called_once()


if __name__ == "__main__":
    unittest.main()
