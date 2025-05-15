"Autonomous Vehicles and Robotics"

import math
import random
import numpy as np
import pygame
from dataclasses import dataclass, field
from typing import List, Tuple, Dict

# Constants
WIDTH = 800
HEIGHT = 600
ROBOT_SIZE = 15
ROBOT_SPEED = 2.0

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

@dataclass
class Robot:
    id: int
    position: Vector2D
    target: Vector2D
    velocity: Vector2D
    color: str
    path: List[Vector2D] = field(default_factory=list)
    artwork_data: np.ndarray = field(default_factory=lambda: np.zeros((100, 100)))
    
    def update(self, dt: float, algorithm: str, environment: 'Environment', other_robots: List['Robot']):
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
    
    def _path_planning(self, dt: float, environment: 'Environment'):
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
    
    def _swarm(self, dt: float, other_robots: List['Robot'], environment: 'Environment'):
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

@dataclass
class Obstacle:
    position: Vector2D
    size: Tuple[float, float]

@dataclass
class Environment:
    id: str
    name: str
    obstacles: List[Obstacle] = field(default_factory=list)
    dimensions: Dict[str, float] = field(default_factory=lambda: {"width": WIDTH, "height": HEIGHT})

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

class Simulation:
    def __init__(self):
        self.state = None
        self.color_palettes = {
            'red': ["#FF6B6B", "#FF8E8E", "#FF5252", "#FF3838"],
            'blue': ["#4D96FF", "#6BA5FF", "#3A86FF", "#2979FF"],
            'green': ["#4CAF50", "#6ECF73", "#388E3C", "#2E7D32"],
            'yellow': ["#FFC107", "#FFD54F", "#FFA000", "#FF8F00"],
            'purple': ["#9C27B0", "#BA68C8", "#7B1FA2", "#6A1B9A"],
            'pink': ["#FF4081", "#FF80AB", "#F50057", "#C51162"]
        }
    
    def initialize(self, robot_count=5, algorithm="random_walk", environment_type="gallery", art_style="blue", speed=1.0):
        # Create environment
        env = self._create_environment(environment_type)
        
        # Create robots
        robots = []
        colors = self.color_palettes.get(art_style, self.color_palettes['blue'])
        
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
        self.state = SimulationState(
            robots=robots,
            environment=env,
            algorithm=algorithm,
            speed=speed
        )
        
        return self.state
    
    def update(self, dt=0.1):
        if not self.state:
            return None
        
        # Update each robot
        for robot in self.state.robots:
            robot.update(
                dt=dt,
                algorithm=self.state.algorithm,
                environment=self.state.environment,
                other_robots=self.state.robots
            )
        
        # Update metrics
        self._update_metrics()
        
        return self.state
    
    def _update_metrics(self):
        # Calculate path efficiency
        path_efficiency = 0
        art_generation = 0
        collision_avoidance = 100  # Assume perfect initially
        
        for robot in self.state.robots:
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
            for obstacle in self.state.environment.obstacles:
                dist = math.sqrt(
                    (robot.position.x - obstacle.position.x)**2 +
                    (robot.position.y - obstacle.position.y)**2
                )
                if dist < (obstacle.size[0] + ROBOT_SIZE) / 2:
                    collision_avoidance -= 5  # Reduce score for each collision
        
        # Normalize metrics
        if len(self.state.robots) > 0:
            path_efficiency /= len(self.state.robots)
            art_generation /= len(self.state.robots)
        
        collision_avoidance = max(0, collision_avoidance)
        
        # Update metrics in state
        self.state.metrics = {
            "path_efficiency": path_efficiency,
            "art_generation": art_generation,
            "collision_avoidance": collision_avoidance,
            "processing_time": 0  # No timing in simplified version
        }
    
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
    
    def export_artwork(self):
        """Export the artwork data in a format suitable for visualization"""
        if not self.state:
            return None
        
        # Combine artwork data from all robots
        artwork_data = []
        for robot in self.state.robots:
            artwork_data.append({
                "artwork": robot.artwork_data.tolist(),  # Convert numpy array to list
                "color": robot.color
            })
        
        return artwork_data

def main():
    """Run a visualization of the simulation using Pygame"""
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("ArtBots Simulation")
    clock = pygame.time.Clock()
    
    # Create and initialize simulation
    simulation = Simulation()
    simulation.initialize(
        robot_count=5,
        algorithm="flocking",
        environment_type="gallery",
        art_style="blue",
        speed=1.0
    )
    
    # Main loop
    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    simulation.state.algorithm = "random_walk"
                elif event.key == pygame.K_2:
                    simulation.state.algorithm = "flocking"
                elif event.key == pygame.K_3:
                    simulation.state.algorithm = "path_planning"
                elif event.key == pygame.K_4:
                    simulation.state.algorithm = "swarm"
                elif event.key == pygame.K_e:
                    # Export artwork
                    print("Artwork exported:", simulation.export_artwork())
        
        # Update simulation state
        simulation.update(dt=0.1)
        
        # Draw background
        screen.fill((242, 242, 242))
        
        # Draw obstacles
        for obstacle in simulation.state.environment.obstacles:
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
        for robot in simulation.state.robots:
            # Draw path
            if len(robot.path) > 1:
                points = [(p.x, p.y) for p in robot.path]
                pygame.draw.lines(screen, pygame.Color(robot.color), False, points, 2)
            
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
            f"Algorithm: {simulation.state.algorithm}",
            f"Path Efficiency: {simulation.state.metrics['path_efficiency']:.1f}%",
            f"Art Generation: {simulation.state.metrics['art_generation']:.1f}%",
            f"Collision Avoidance: {simulation.state.metrics['collision_avoidance']:.1f}%",
            f"Controls: 1-4 change algorithm, E exports artwork"
        ]
        
        for i, text in enumerate(metrics_text):
            surface = font.render(text, True, (0, 0, 0))
            screen.blit(surface, (10, 10 + i * 25))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()

if __name__ == "__main__":
    main()