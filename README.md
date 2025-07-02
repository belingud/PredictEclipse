# PredictEclipse

A Python tool to predict solar and lunar eclipses with high precision using JPL ephemeris data.

## Features

- Predict solar and lunar eclipses for any time range
- Display detailed information including:
  - Eclipse type (Total, Partial, Annular, Penumbral)
  - Earth-Moon and Earth-Sun distances
  - Apparent diameters of the Moon and Sun
  - Node distance and angular measurements
  - Size ratio for solar eclipses
- Automatic selection of the best ephemeris file based on the requested time range
- Dynamic fetching of available ephemeris files from JPL's website
- Support for local timezone conversion

## Installation

1. Clone this repository:
```bash
git clone https://github.com/belingud/PredictEclipse.git
cd PredictEclipse
```

2. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Eclipse Prediction

To predict solar eclipses for the next two years:
```bash
python predict_elipse.py --type solar
```

To predict lunar eclipses for the next two years:
```bash
python predict_elipse.py --type lunar
```

### Custom Date Range

To predict eclipses for a specific date range:
```bash
python predict_elipse.py --type solar --start 2024-01-01 --end 2030-12-31
```

### Specifying an Ephemeris File

You can manually specify which JPL ephemeris file to use:
```bash
python predict_elipse.py --type lunar --ephemeris de440.bsp
# or use a url
python predict_elipse.py --eph https://naif.jpl.nasa.gov/pub/naif/generic_kernels/spk/planets/de432s.bsp --type lunar
```

### Specifying Skyfield Dir

You can specify skyfield data dir to save and use:

```bash
python predict_elipse.py --type lunar --ephemeris de440.bsp --data-dir ~/skyfield-data
```

### Listing Available Ephemeris Files

To see all available ephemeris files (fetched from JPL's website) and their coverage:
```bash
python predict_elipse.py --list-eph
```

## Command Line Options

| Option                 | Description                                                        |
| ---------------------- | ------------------------------------------------------------------ |
| `--type {solar,lunar}` | Required. Specify the type of eclipse to predict                   |
| `--start YYYY-MM-DD`   | Start date (default: today)                                        |
| `--end YYYY-MM-DD`     | End date (default: two years from start date)                      |
| `--ephemeris FILE`     | Ephemeris file to use (default: auto-selected based on date range) |
| `--data-dir`           | Specific skyfield data save dir (default: current dir)             |
| `--list-eph`           | List all available ephemeris files and their coverage              |

## Understanding Ephemeris Files

JPL ephemeris files (BSP format) contain high-precision planetary position data:

- `de421.bsp`: Common, small file (17MB), good for predictions between 1900-2050
- `de430.bsp`: Higher precision (115MB), covers 1550-2650
- `de440.bsp`: Current standard JPL ephemeris (114MB), covers 1550-2650
- `de440s.bsp`: Ultra-long time range version (32MB), covers 13000 BCE to 17000 CE
- `de441.bsp`: Extended time range with high precision (770MB)

The script automatically selects the best ephemeris file based on your date range, balancing precision, coverage, and file size.

## How It Works

The script uses the Skyfield astronomy library to calculate the positions of the Earth, Moon, and Sun. It then:

1. Identifies full moons (for lunar eclipses) or new moons (for solar eclipses)
2. Calculates the Sun-Earth-Moon angle to determine potential eclipses
3. Calculates Earth-Moon and Earth-Sun distances
4. Determines eclipse type based on apparent diameters and node distance
5. Converts UTC to local time for better readability

## Requirements

- Python 3.9+
- skyfield>=1.41.0
- numpy>=1.23.0
- requests>=2.27.0 (for online ephemeris lookup)
- beautifulsoup4>=4.10.0 (for online ephemeris lookup)

## License

[MIT License](LICENSE)

## Acknowledgements

- Uses NASA JPL's NAIF ephemeris data
- Built with the [Skyfield](https://rhodesmill.org/skyfield/) astronomy library
