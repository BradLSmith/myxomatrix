# The Myxomatrix

An artificial life simulation that evolves population level complexity without predefinition of a 'goal' state via a fitness function or convoluted initial conditions. 

## Features

- **Evolved Neural Networks**: Agents use spiking neural networks that evolve over time
- **Ecological Simulation**: Plants, growth regions, and resource dynamics
- **Species Management**: Automatic speciation based on genetic compatibility
- **Video Recording**: Record your simulations as MP4 videos
- **Interactive Setup**: GUI-based environment configuration
- **Data Analysis**: Automatic generation of simulation statistics and plots

## Installation

### Prerequisites

- **Python 3.7+** (recommended: Python 3.8 or higher)
- **FFmpeg** (for video recording functionality)

### Step 1: Clone the Repository

```bash
git clone https://github.com/yourusername/myxomatrix.git
cd myxomatrix
```

### Step 2: Install Python Dependencies

#### Option A: Using pip
```bash
pip install -r requirements.txt
```

#### Option B: Install manually
```bash
pip install pygame pygame-gui numpy scipy matplotlib
```

### Step 3: Install FFmpeg

FFmpeg is required for the video recording feature. Choose your platform:

#### Windows

**Option 1: Using Chocolatey (Recommended)**
```powershell
# Install Chocolatey first if you don't have it
# Then run:
choco install ffmpeg
```

**Option 2: Manual Installation**
1. Download FFmpeg from [https://ffmpeg.org/download.html#build-windows](https://ffmpeg.org/download.html#build-windows)
2. Extract the archive to a folder (e.g., `C:\ffmpeg`)
3. Add the `bin` folder to your PATH:
   - Open "Environment Variables" in System Properties
   - Edit the `PATH` variable
   - Add `C:\ffmpeg\bin` (or wherever you extracted it)
   - Restart your command prompt/IDE

#### macOS

**Option 1: Using Homebrew (Recommended)**
```bash
# Install Homebrew first if you don't have it: https://brew.sh
brew install ffmpeg
```

**Option 2: Using MacPorts**
```bash
sudo port install ffmpeg
```

#### Linux (Ubuntu/Debian)

```bash
sudo apt update
sudo apt install ffmpeg
```

#### Linux (CentOS/RHEL/Fedora)

```bash
# CentOS/RHEL
sudo yum install ffmpeg

# Fedora
sudo dnf install ffmpeg
```

### Step 4: Verify FFmpeg Installation

Open a terminal/command prompt and run:
```bash
ffmpeg -version
```

You should see version information. If you get a "command not found" error, FFmpeg is not properly installed or not in your PATH.

## Usage

### Running the Simulation

```bash
python main.py
```

The program will:
1. Show a logo screen
2. Open the environment setup interface
3. Allow you to configure the simulation parameters
4. Start the simulation

### Controls

- **R** - Restart simulation with a new random seed
- **Q** - Quit to setup interface
- **V** - Start/stop video recording
- **C** - Cancel ongoing video recording
- **ESC/Close** - Exit the program

### Setup Interface

The setup interface allows you to configure:
- **Environment size** (width/height)
- **Growth regions** (areas where plants spawn)
- **Walls** (barriers for agents)
- **Hyperparameters** (mutation rates, costs, etc.)
- **Agent mobility** (stationary vs. moving agents)
- **Random seed** (for reproducible results)

#### Creating Growth Regions
1. Click and drag on the grid to create a rectangular region
2. Click on a region to select and modify its properties
3. Right-click on a region to delete it
4. Configure plant density, growth rates, and initial agent count

### Video Recording

1. Press **V** during simulation to start recording
2. Press **V** again to stop and begin video processing
3. Videos are saved in the `simulation_recordings/` folder
4. Processing includes a logo intro and fade effects

**Note**: Video recording requires significant disk space for temporary frames.

### Data Analysis

After each simulation run, the program automatically:
- Saves CSV data to `simulation_results/`
- Generates plots showing population dynamics, neural network evolution, and behavioral statistics
- Creates timestamped folders for each run

## Configuration Files

The program saves your setup configuration to `setup_config.json`, which you can:
- Save and load different experimental setups
- Share configurations with other researchers
- Version control your experimental parameters

## System Requirements

### Minimum Requirements
- **RAM**: 2GB available
- **Disk Space**: 1GB free (more for video recordings)
- **CPU**: Any modern multi-core processor
- **Display**: 1400x860 minimum resolution for setup interface

### Recommended for Large Simulations
- **RAM**: 8GB+ available
- **Disk Space**: 10GB+ free
- **CPU**: Modern multi-core processor (4+ cores)
- **SSD**: For better performance with large simulations

## Troubleshooting

### Common Issues

**"ModuleNotFoundError: No module named 'pygame'"**
- Solution: Install the required packages with `pip install -r requirements.txt`

**"ffmpeg: command not found" or video recording fails**
- Solution: Install FFmpeg and ensure it's in your system PATH
- Test with `ffmpeg -version` in terminal

**Simulation runs very slowly**
- Try reducing the environment size
- Reduce the number of agents
- Close other applications to free up RAM

**Setup interface appears corrupted or unusable**
- Ensure your display resolution is at least 1400x860
- Try running on a different monitor if using multiple displays

**Out of memory errors**
- Reduce environment size and agent population
- Restart the program periodically for very long runs

### Platform-Specific Notes

**Windows**
- Some antivirus software may flag the executable - this is a false positive
- If using Windows 7, ensure you have the latest updates

**macOS**
- On macOS Monterey+, you may need to grant permission for screen recording
- If using Apple Silicon (M1/M2), all dependencies should work natively

**Linux**
- Ensure you have the necessary graphics drivers installed
- On headless systems, you'll need to set up a virtual display

## Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this software in your research, please cite:

```
[Your Name] (2025). The Myxomatrix: Evolving Population-level Complexity in Darwinian
Artificial Life. GitHub repository: https://github.com/yourusername/myxomatrix
```

## Acknowledgments

This project builds upon concepts from:
- Artificial Life and Evolutionary Computation
- Spiking Neural Networks
- Ecological Modeling
- NEAT (NeuroEvolution of Augmenting Topologies)

---

For questions, issues, or collaboration opportunities, please open an issue on GitHub or contact [brad.luke.smith@outlook.com].
