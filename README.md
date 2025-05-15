# autonomousvehicle
#  Autonomous vehicles Robot Art Simulation

A simplified, interactive Python simulation using autonomous robots that create generative art while navigating a 2D environment. Inspired by swarm intelligence and robotics algorithms, this project is suitable for learning or demonstration purposes in college-level robotics, AI, or generative art.

---

## 🖼️ Features

- **Multiple Movement Algorithms**:
  - `random_walk`: Robots explore randomly.
  - `flocking`: Boids-like swarm behavior.
  - `path_planning`: Simple target-seeking behavior.
  - `swarm`: Cooperative, center-attracted group dynamics.
- **Interactive Visualization** with Pygame
- **Real-time Metric Evaluation**:
  - Path Efficiency
  - Art Generation (canvas coverage)
  - Collision Avoidance
- **Environment Types**:
  - Art Gallery
  - Studio
  - Museum
  - Outdoors
- **Art Exporting**: Save artwork data from all robots.

---

## 🎮 Controls

| Key | Function                |
|-----|-------------------------|
| 1   | Switch to Random Walk   |
| 2   | Switch to Flocking      |
| 3   | Switch to Path Planning |
| 4   | Switch to Swarm Mode    |
| E   | Export artwork data     |
| ESC / Close | Exit the simulation |

---

## 🛠️ Installation & Running

### Requirements

- Python 3.7+
- `pygame`
- `numpy`

### Install Dependencies

```bash
pip install pygame numpy
