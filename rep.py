"""
ArtBots: Autonomous Robot Art Simulation
A college project demonstrating robotic swarm behavior and generative art

Author: Your Name
License: MIT
"""

import os
import math
import json
import random
import uuid
import datetime
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Any
import pygame
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit

# Custom JSON encoder to handle numpy arrays
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        if isinstance(obj, Vector2D):
            return {"x": obj.x, "y": obj.y}
        return super(NumpyEncoder, self).default(obj)

# Constants
WIDTH = 800
HEIGHT = 600
ROBOT_SIZE = 15
ROBOT_SPEED = 2.0

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'artbots-secret-key'
app.json_encoder = NumpyEncoder
socketio = SocketIO(app, cors_allowed_origins="*", json=json)

# Data models
@dataclass
class Vector2D:
    x: float
    y: float
    
    def distance_to(self, other: 'Vector2D') -> float:
        return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2)
    
    def normalize(self) -> 'Vector2D':
        mag = math.sqrt(self.x ** 2 + self.y ** 2)
        if mag > 0:
            return Vector2D(self.x / mag, self.y / mag)
        return Vector2D(0, 0)
    
    def to_dict(self):
        return {"x": self.x, "y": self.y}

@dataclass
class Robot:
    id: int
    position: Vector2D
    target: Vector2D
    velocity: Vector2D
    color: str
    path: List[Vector2D] = field(default_factory=list)
    artwork_data: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))
    
    def update(self, dt: float, algorithm: str, environment: Any, other_robots: List['Robot']):
        if algorithm == "random_walk":
            self._random_walk(dt)
        elif algorithm == "flocking":
            self._flocking(dt, other_robots)
        elif algorithm == "path_planning":
            self._path_planning(dt, environment)
        elif algorithm == "swarm":
            self._swarm(dt, other_robots, environment)
        
        # Update position based on velocity
        self.position.x += self.velocity.x * dt
        self.position.y += self.velocity.y * dt
        
        # Constrain to environment boundaries
        self.position.x = max(0, min(WIDTH, self.position.x))
        self.position.y = max(0, min(HEIGHT, self.position.y))
        
        # Update path
        self.path.append(Vector2D(self.position.x, self.position.y))
        if len(self.path) > 100:
            self.path.pop(0)
        
        # Update artwork data by marking pixels where robot passes
        art_x = int(self.position.x / WIDTH * 99)
        art_y = int(self.position.y / HEIGHT * 99)
        self.artwork_data[art_y, art_x] += 1
    
    def _random_walk(self, dt: float):
        # Simple random walk
        if random.random() < 0.05 or self.position.distance_to(self.target) < 10:
            self.target = Vector2D(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT)
            )
        
        direction = Vector2D(
            self.target.x - self.position.x,
            self.target.y - self.position.y
        )
        normalized = direction.normalize()
        self.velocity.x = normalized.x * ROBOT_SPEED
        self.velocity.y = normalized.y * ROBOT_SPEED
    
    def _flocking(self, dt: float, other_robots: List['Robot']):
        # Boids-like flocking algorithm
        separation = Vector2D(0, 0)
        alignment = Vector2D(0, 0)
        cohesion = Vector2D(0, 0)
        
        neighbor_count = 0
        for robot in other_robots:
            if robot.id != self.id:
                distance = self.position.distance_to(robot.position)
                
                if distance < 100:  # Neighborhood radius
                    neighbor_count += 1
                    
                    # Separation - move away from close neighbors
                    if distance < 30:
                        diff = Vector2D(
                            self.position.x - robot.position.x,
                            self.position.y - robot.position.y
                        )
                        separation.x += diff.x / max(distance, 0.1)
                        separation.y += diff.y / max(distance, 0.1)
                    
                    # Alignment - align with neighbors' velocities
                    alignment.x += robot.velocity.x
                    alignment.y += robot.velocity.y
                    
                    # Cohesion - move toward center of neighbors
                    cohesion.x += robot.position.x
                    cohesion.y += robot.position.y
        
        if neighbor_count > 0:
            # Normalize and apply weights
            cohesion.x = (cohesion.x / neighbor_count - self.position.x) * 0.01
            cohesion.y = (cohesion.y / neighbor_count - self.position.y) * 0.01
            
            alignment.x = (alignment.x / neighbor_count) * 0.1
            alignment.y = (alignment.y / neighbor_count) * 0.1
            
            sep_norm = math.sqrt(separation.x ** 2 + separation.y ** 2)
            if sep_norm > 0:
                separation.x = (separation.x / sep_norm) * 0.2
                separation.y = (separation.y / sep_norm) * 0.2
            
            # Apply forces
            self.velocity.x += separation.x + alignment.x + cohesion.x
            self.velocity.y += separation.y + alignment.y + cohesion.y
            
            # Normalize velocity to maintain consistent speed
            vel_norm = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2)
            if vel_norm > 0:
                self.velocity.x = (self.velocity.x / vel_norm) * ROBOT_SPEED
                self.velocity.y = (self.velocity.y / vel_norm) * ROBOT_SPEED
        else:
            # If no neighbors, use random walk
            self._random_walk(dt)
    
    def _path_planning(self, dt: float, environment: Any):
        # Simple path planning with target points
        if self.position.distance_to(self.target) < 10:
            # Set new target when reaching current one
            self.target = Vector2D(
                random.uniform(0, WIDTH),
                random.uniform(0, HEIGHT)
            )
        
        # Move toward target
        direction = Vector2D(
            self.target.x - self.position.x,
            self.target.y - self.position.y
        )
        normalized = direction.normalize()
        self.velocity.x = normalized.x * ROBOT_SPEED
        self.velocity.y = normalized.y * ROBOT_SPEED
    
    def _swarm(self, dt: float, other_robots: List['Robot'], environment: Any):
        # Swarm intelligence with attraction to center
        center_x = 0
        center_y = 0
        for robot in other_robots:
            center_x += robot.position.x
            center_y += robot.position.y
        
        center_x /= len(other_robots)
        center_y /= len(other_robots)
        
        # Attraction to center
        to_center = Vector2D(
            center_x - self.position.x,
            center_y - self.position.y
        )
        to_center_norm = math.sqrt(to_center.x ** 2 + to_center.y ** 2)
        if to_center_norm > 0:
            to_center.x = (to_center.x / to_center_norm) * 0.02
            to_center.y = (to_center.y / to_center_norm) * 0.02
        
        # Repulsion from other robots
        repulsion = Vector2D(0, 0)
        for robot in other_robots:
            if robot.id != self.id:
                distance = self.position.distance_to(robot.position)
                if distance < 40 and distance > 0:
                    repulsion.x += (self.position.x - robot.position.x) / distance * 0.1
                    repulsion.y += (self.position.y - robot.position.y) / distance * 0.1
        
        # Apply forces
        self.velocity.x += to_center.x + repulsion.x + random.uniform(-0.1, 0.1)
        self.velocity.y += to_center.y + repulsion.y + random.uniform(-0.1, 0.1)
        
        # Normalize velocity
        vel_norm = math.sqrt(self.velocity.x ** 2 + self.velocity.y ** 2)
        if vel_norm > 0:
            self.velocity.x = (self.velocity.x / vel_norm) * ROBOT_SPEED
            self.velocity.y = (self.velocity.y / vel_norm) * ROBOT_SPEED
    
    def to_dict(self):
        return {
            "id": self.id,
            "position": self.position.to_dict(),
            "target": self.target.to_dict(),
            "velocity": self.velocity.to_dict(),
            "color": self.color,
            "path": [p.to_dict() for p in self.path],
            "artwork_data": self.artwork_data.tolist()
        }

@dataclass
class Obstacle:
    position: Vector2D
    size: Tuple[float, float]
    
    def to_dict(self):
        return {
            "position": self.position.to_dict(),
            "size": self.size
        }

@dataclass
class Environment:
    id: str
    name: str
    obstacles: List[Obstacle] = field(default_factory=list)
    dimensions: Dict[str, float] = field(default_factory=lambda: {"width": WIDTH, "height": HEIGHT})
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "obstacles": [o.to_dict() for o in self.obstacles],
            "dimensions": self.dimensions
        }

@dataclass
class Artwork:
    id: int
    title: str
    description: str
    thumbnail_url: str
    date: datetime.datetime
    algorithm: str
    environment: str
    settings: Dict[str, Any]
    
    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "thumbnail_url": self.thumbnail_url,
            "date": self.date.isoformat(),
            "algorithm": self.algorithm,
            "environment": self.environment,
            "settings": self.settings
        }

@dataclass
class SimulationSettings:
    id: int
    name: str
    robot_count: int
    speed: float
    algorithm: str
    environment: str
    art_style: str
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "robot_count": self.robot_count,
            "speed": self.speed,
            "algorithm": self.algorithm,
            "environment": self.environment,
            "art_style": self.art_style
        }

@dataclass
class SimulationState:
    robots: List[Robot]
    environment: Environment
    algorithm: str
    speed: float
    metrics: Dict[str, float] = field(default_factory=lambda: {
        "path_efficiency": 0,
        "art_generation": 0,
        "collision_avoidance": 0,
        "processing_time": 0
    })
    
    def to_dict(self):
        return {
            "robots": [r.to_dict() for r in self.robots],
            "environment": self.environment.to_dict(),
            "algorithm": self.algorithm,
            "speed": self.speed,
            "metrics": self.metrics
        }

# Simulation manager
class SimulationManager:
    def __init__(self):
        self.active_simulations = {}
        self.artworks = []
        self.settings_presets = []
        self.next_artwork_id = 1
        
        # Add some sample artworks
        self._add_sample_artworks()
    
    def _add_sample_artworks(self):
        self.artworks.append(Artwork(
            id=self.next_artwork_id,
            title="Algorithmic Impressions",
            description="Created by autonomous robots using a flocking behavior algorithm with a custom color palette.",
            thumbnail_url="/static/images/artwork1.jpg",
            date=datetime.datetime.now() - datetime.timedelta(days=30),
            algorithm="flocking",
            environment="gallery",
            settings={"robots": 5, "speed": 7, "colors": ["#FF6B6B", "#4D96FF"]}
        ))
        self.next_artwork_id += 1
        
        self.artworks.append(Artwork(
            id=self.next_artwork_id,
            title="Synthetic Dreamscapes",
            description="Generated through path-planning algorithms that translate movement patterns into visual elements.",
            thumbnail_url="/static/images/artwork2.jpg",
            date=datetime.datetime.now() - datetime.timedelta(days=15),
            algorithm="path_planning",
            environment="museum",
            settings={"robots": 3, "speed": 5, "colors": ["#4CAF50", "#FFC107"]}
        ))
        self.next_artwork_id += 1
        
        self.artworks.append(Artwork(
            id=self.next_artwork_id,
            title="Mechanical Perspectives",
            description="This piece demonstrates emergent creativity when multiple robots collaborate on canvas.",
            thumbnail_url="/static/images/artwork3.jpg",
            date=datetime.datetime.now() - datetime.timedelta(days=7),
            algorithm="swarm",
            environment="studio",
            settings={"robots": 8, "speed": 6, "colors": ["#9C27B0", "#FF4081"]}
        ))
        self.next_artwork_id += 1
    
    def start_simulation(self, settings):
        sim_id = str(uuid.uuid4())
        
        # Create environment
        env_type = settings.get('environment', 'gallery')
        environment = self._create_environment(env_type)
        
        # Create robots
        robots = []
        robot_count = settings.get('robot_count', 5)
        art_style = settings.get('art_style', 'blue')
        
        # Color palettes
        color_palettes = {
            'red': ["#FF6B6B", "#FF8E8E", "#FF5252", "#FF3838"],
            'blue': ["#4D96FF", "#6BA5FF", "#3A86FF", "#2979FF"],
            'green': ["#4CAF50", "#6ECF73", "#388E3C", "#2E7D32"],
            'yellow': ["#FFC107", "#FFD54F", "#FFA000", "#FF8F00"],
            'purple': ["#9C27B0", "#BA68C8", "#7B1FA2", "#6A1B9A"],
            'pink': ["#FF4081", "#FF80AB", "#F50057", "#C51162"],
        }
        
        colors = color_palettes.get(art_style, color_palettes['blue'])
        
        for i in range(robot_count):
            robots.append(Robot(
                id=i,
                position=Vector2D(
                    random.uniform(50, WIDTH-50),
                    random.uniform(50, HEIGHT-50)
                ),
                target=Vector2D(
                    random.uniform(0, WIDTH),
                    random.uniform(0, HEIGHT)
                ),
                velocity=Vector2D(0, 0),
                color=colors[i % len(colors)]
            ))
        
        # Create simulation state
        sim_state = SimulationState(
            robots=robots,
            environment=environment,
            algorithm=settings.get('algorithm', 'random_walk'),
            speed=settings.get('speed', 1.0)
        )
        
        self.active_simulations[sim_id] = sim_state
        return sim_id, sim_state
    
    def update_simulation(self, sim_id):
        if sim_id not in self.active_simulations:
            return None
        
        start_time = datetime.datetime.now()
        state = self.active_simulations[sim_id]
        
        # Update each robot
        for robot in state.robots:
            robot.update(
                dt=0.1,
                algorithm=state.algorithm,
                environment=state.environment,
                other_robots=state.robots
            )
        
        # Update metrics
        processing_time = (datetime.datetime.now() - start_time).total_seconds() * 1000
        
        # Calculate path efficiency
        path_efficiency = 0
        art_generation = 0
        collision_avoidance = 100  # Assume perfect initially
        
        for robot in state.robots:
            # Art generation - percentage of canvas covered
            art_points = np.count_nonzero(robot.artwork_data)
            art_generation += (art_points / (100 * 100)) * 100
            
            # Path efficiency (approximation)
            if len(robot.path) > 2:
                start = robot.path[0]
                end = robot.path[-1]
                straight_line = math.sqrt((end.x - start.x)**2 + (end.y - start.y)**2)
                
                actual_path = 0
                for i in range(1, len(robot.path)):
                    actual_path += math.sqrt(
                        (robot.path[i].x - robot.path[i-1].x)**2 + 
                        (robot.path[i].y - robot.path[i-1].y)**2
                    )
                
                if actual_path > 0:
                    path_efficiency += (straight_line / actual_path) * 100
                
            # Check for collisions with obstacles
            for obstacle in state.environment.obstacles:
                dist = math.sqrt(
                    (robot.position.x - obstacle.position.x)**2 +
                    (robot.position.y - obstacle.position.y)**2
                )
                if dist < (obstacle.size[0] + ROBOT_SIZE) / 2:
                    collision_avoidance -= 5  # Reduce score for each collision
        
        # Normalize metrics
        if len(state.robots) > 0:
            path_efficiency /= len(state.robots)
            art_generation /= len(state.robots)
        
        collision_avoidance = max(0, collision_avoidance)
        
        # Update metrics in state
        state.metrics = {
            "path_efficiency": path_efficiency,
            "art_generation": art_generation,
            "collision_avoidance": collision_avoidance,
            "processing_time": processing_time
        }
        
        return state
    
    def export_artwork(self, sim_id):
        if sim_id not in self.active_simulations:
            return None
        
        state = self.active_simulations[sim_id]
        
        # Combine artwork data from all robots
        artwork_data = []
        for robot in state.robots:
            artwork_data.append({
                "artwork": robot.artwork_data.tolist(),  # Convert numpy array to list
                "color": robot.color
            })
        
        # Create new artwork entry
        artwork = Artwork(
            id=self.next_artwork_id,
            title=f"Generated Artwork {self.next_artwork_id}",
            description=f"Created using {state.algorithm} algorithm with {len(state.robots)} robots",
            thumbnail_url="/static/images/generated.jpg",  # This would be generated in a real app
            date=datetime.datetime.now(),
            algorithm=state.algorithm,
            environment=state.environment.id,
            settings={
                "robots": len(state.robots),
                "speed": state.speed,
                "colors": [robot.color for robot in state.robots]
            }
        )
        
        self.artworks.append(artwork)
        self.next_artwork_id += 1
        
        return artwork_data
    
    def _create_environment(self, env_type):
        # Create different environment layouts
        if env_type == "gallery":
            obstacles = [
                Obstacle(Vector2D(WIDTH/2, HEIGHT/2), (50, 50)),
                Obstacle(Vector2D(WIDTH/4, HEIGHT/4), (30, 30)),
                Obstacle(Vector2D(3*WIDTH/4, 3*HEIGHT/4), (30, 30))
            ]
            return Environment("gallery", "Art Gallery", obstacles)
        
        elif env_type == "studio":
            obstacles = [
                Obstacle(Vector2D(WIDTH/3, HEIGHT/3), (20, 20)),
                Obstacle(Vector2D(2*WIDTH/3, 2*HEIGHT/3), (20, 20))
            ]
            return Environment("studio", "Art Studio", obstacles)
        
        elif env_type == "museum":
            obstacles = [
                Obstacle(Vector2D(WIDTH/5, HEIGHT/5), (40, 40)),
                Obstacle(Vector2D(WIDTH/2, HEIGHT/2), (60, 60)),
                Obstacle(Vector2D(4*WIDTH/5, 4*HEIGHT/5), (40, 40))
            ]
            return Environment("museum", "Museum", obstacles)
        
        elif env_type == "outdoors":
            obstacles = [
                Obstacle(Vector2D(WIDTH/4, HEIGHT/4), (50, 50)),
                Obstacle(Vector2D(3*WIDTH/4, 3*HEIGHT/4), (50, 50)),
                Obstacle(Vector2D(WIDTH/4, 3*HEIGHT/4), (50, 50)),
                Obstacle(Vector2D(3*WIDTH/4, HEIGHT/4), (50, 50))
            ]
            return Environment("outdoors", "Outdoors", obstacles)
        
        # Default to gallery
        return Environment("gallery", "Art Gallery", [])

# Create simulation manager instance
simulation_manager = SimulationManager()

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/artworks')
def get_artworks():
    return jsonify([artwork.to_dict() for artwork in simulation_manager.artworks])

@app.route('/api/artworks/<int:artwork_id>')
def get_artwork(artwork_id):
    for artwork in simulation_manager.artworks:
        if artwork.id == artwork_id:
            return jsonify(artwork.to_dict())
    return jsonify({"error": "Artwork not found"}), 404

# Socket.IO events
@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('startSimulation')
def handle_start_simulation(data):
    settings = data.get('settings', {})
    sim_id, sim_state = simulation_manager.start_simulation(settings)
    
    # Convert the simulation state to dict for JSON serialization
    socketio.emit('simulationStart', {
        'simulationId': sim_id,
        'state': sim_state.to_dict()
    }, room=request.sid)
    
    # Start sending updates
    def send_updates():
        while True:
            updated_state = simulation_manager.update_simulation(sim_id)
            if updated_state:
                socketio.emit('simulationUpdate', {
                    'simulationId': sim_id,
                    'state': updated_state.to_dict()
                })
                socketio.sleep(0.1)
            else:
                break
    
    socketio.start_background_task(send_updates)

@socketio.on('updateSettings')
def handle_update_settings(data):
    sim_id = data.get('simulationId')
    settings = data.get('settings', {})
    
    if sim_id in simulation_manager.active_simulations:
        state = simulation_manager.active_simulations[sim_id]
        
        # Update settings
        if 'algorithm' in settings:
            state.algorithm = settings['algorithm']
        if 'speed' in settings:
            state.speed = settings['speed']
        if 'environment' in settings:
            state.environment = simulation_manager._create_environment(settings['environment'])
        if 'robot_count' in settings:
            current_count = len(state.robots)
            new_count = settings['robot_count']
            
            if new_count > current_count:
                # Add more robots
                for i in range(current_count, new_count):
                    state.robots.append(Robot(
                        id=i,
                        position=Vector2D(
                            random.uniform(50, WIDTH-50),
                            random.uniform(50, HEIGHT-50)
                        ),
                        target=Vector2D(
                            random.uniform(0, WIDTH),
                            random.uniform(0, HEIGHT)
                        ),
                        velocity=Vector2D(0, 0),
                        color=state.robots[0].color if state.robots else "#4D96FF"
                    ))
            elif new_count < current_count:
                # Remove robots
                state.robots = state.robots[:new_count]
        
        if 'art_style' in settings:
            # Update robot colors
            color_palettes = {
                'red': ["#FF6B6B", "#FF8E8E", "#FF5252", "#FF3838"],
                'blue': ["#4D96FF", "#6BA5FF", "#3A86FF", "#2979FF"],
                'green': ["#4CAF50", "#6ECF73", "#388E3C", "#2E7D32"],
                'yellow': ["#FFC107", "#FFD54F", "#FFA000", "#FF8F00"],
                'purple': ["#9C27B0", "#BA68C8", "#7B1FA2", "#6A1B9A"],
                'pink': ["#FF4081", "#FF80AB", "#F50057", "#C51162"],
            }
            
            colors = color_palettes.get(settings['art_style'], color_palettes['blue'])
            for i, robot in enumerate(state.robots):
                robot.color = colors[i % len(colors)]

@socketio.on('stopSimulation')
def handle_stop_simulation(data):
    sim_id = data.get('simulationId')
    if sim_id in simulation_manager.active_simulations:
        del simulation_manager.active_simulations[sim_id]

@socketio.on('exportArtwork')
def handle_export_artwork(data):
    sim_id = data.get('simulationId')
    artwork_data = simulation_manager.export_artwork(sim_id)
    
    if artwork_data:
        emit('artworkData', {
            'data': artwork_data
        })

# Create necessary directories
def setup_static_dirs():
    os.makedirs('static/images', exist_ok=True)
    
    # You might want to add some placeholder images for artworks
    # This would be replaced with actual generated content in a full application

# Visualization using Pygame (for local testing)
def start_visualization():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ArtBots Simulation")
    clock = pygame.time.Clock()
    
    # Create a simple simulation for visualization
    settings = {
        'robot_count': 5,
        'algorithm': 'flocking',
        'environment': 'gallery',
        'art_style': 'blue',
        'speed': 1.0
    }
    
    sim_id, _ = simulation_manager.start_simulation(settings)
    
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        # Update simulation
        state = simulation_manager.update_simulation(sim_id)
        
        # Draw background
        screen.fill((242, 242, 242))
        
        # Draw obstacles
        for obstacle in state.environment.obstacles:
            pygame.draw.rect(
                screen,
                (200, 200, 200),
                (
                    obstacle.position.x - obstacle.size[0]/2, 
                    obstacle.position.y - obstacle.size[1]/2,
                    obstacle.size[0],
                    obstacle.size[1]
                )
            )
        
        # Draw robots
        for robot in state.robots:
            # Draw path
            if len(robot.path) > 1:
                points = [(p.x, p.y) for p in robot.path]
                color = pygame.Color(robot.color)
                color.a = 100  # Semi-transparent
                pygame.draw.lines(screen, color, False, points, 2)
            
            # Draw robot
            pygame.draw.circle(
                screen,
                pygame.Color(robot.color),
                (int(robot.position.x), int(robot.position.y)),
                ROBOT_SIZE
            )
            
            # Draw direction indicator
            direction_x = robot.position.x + robot.velocity.x * 10
            direction_y = robot.position.y + robot.velocity.y * 10
            pygame.draw.line(
                screen,
                (0, 0, 0),
                (int(robot.position.x), int(robot.position.y)),
                (int(direction_x), int(direction_y)),
                2
            )
        
        # Draw metrics
        font = pygame.font.SysFont(None, 24)
        metrics_text = [
            f"Algorithm: {state.algorithm}",
            f"Path Efficiency: {state.metrics['path_efficiency']:.1f}%",
            f"Art Generation: {state.metrics['art_generation']:.1f}%",
            f"Collision Avoidance: {state.metrics['collision_avoidance']:.1f}%",
            f"Processing Time: {state.metrics['processing_time']:.1f}ms"
        ]
        
        for i, text in enumerate(metrics_text):
            surface = font.render(text, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(30)
    
    pygame.quit()

if __name__ == '__main__':
    # Setup static directories
    setup_static_dirs()
    
    # For visualization testing (uncomment to use)
    # vis_thread = threading.Thread(target=start_visualization)
    # vis_thread.daemon = True
    # vis_thread.start()
    
    # Start Flask-SocketIO server
    socketio.run(app, host='0.0.0.0', port=5000, debug=True)