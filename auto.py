import pygame
import random

# Init
pygame.init()
WIDTH, HEIGHT = 400, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Home to Office")

clock = pygame.time.Clock()
FPS = 60

# Colors
WHITE = (255, 255, 255)
GRAY = (50, 50, 50)
RED = (200, 0, 0)
GREEN = (0, 200, 0)
YELLOW = (255, 255, 0)
BLUE = (0, 0, 255)

# Car sizes
CAR_WIDTH = 40
CAR_HEIGHT = 70

# Player (Self-driving car)
player = pygame.Rect(WIDTH // 2 - CAR_WIDTH // 2, HEIGHT - CAR_HEIGHT - 10, CAR_WIDTH, CAR_HEIGHT)

# Traffic vehicles
traffic = []
SPAWN_EVENT = pygame.USEREVENT + 1
pygame.time.set_timer(SPAWN_EVENT, 1000)

# Destination
office_y = 30
office_rect = pygame.Rect(0, 0, WIDTH, 50)

def draw_road():
    screen.fill(GRAY)
    for y in range(0, HEIGHT, 40):
        pygame.draw.rect(screen, WHITE, (WIDTH // 2 - 5, y, 10, 20))
    pygame.draw.rect(screen, BLUE, office_rect)
    font = pygame.font.SysFont(None, 30)
    label = font.render("OFFICE", True, WHITE)
    screen.blit(label, (WIDTH // 2 - 30, 10))

def spawn_traffic():
    lane = random.choice([WIDTH // 4 - CAR_WIDTH // 2,
                          WIDTH // 2 - CAR_WIDTH // 2,
                          3 * WIDTH // 4 - CAR_WIDTH // 2])
    car = pygame.Rect(lane, -CAR_HEIGHT, CAR_WIDTH, CAR_HEIGHT)
    traffic.append(car)

def move_traffic():
    for car in traffic:
        car.y += 5
    # Remove off-screen cars
    return [car for car in traffic if car.y < HEIGHT + CAR_HEIGHT]

def check_collision():
    for car in traffic:
        if player.colliderect(car):
            return True
    return False

def auto_avoid():
    """Simple automatic left/right move to avoid cars."""
    for car in traffic:
        if abs(car.y - player.y) < CAR_HEIGHT and abs(car.x - player.x) < CAR_WIDTH:
            if player.x > WIDTH // 2:
                player.x -= 50  # move left
            else:
                player.x += 50  # move right

running = True
won = False
while running:
    clock.tick(FPS)
    draw_road()

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == SPAWN_EVENT:
            spawn_traffic()

    # Move traffic and check collision
    traffic = move_traffic()
    for car in traffic:
        pygame.draw.rect(screen, RED, car)

    # Auto or manual driving
    keys = pygame.key.get_pressed()
    if keys[pygame.K_LEFT] and player.x > 10:
        player.x -= 5
    elif keys[pygame.K_RIGHT] and player.x < WIDTH - CAR_WIDTH - 10:
        player.x += 5
    else:
        auto_avoid()

    player.y -= 2  # constant upward movement (toward office)

    # Check win
    if player.top <= office_y:
        won = True
        running = False

    if check_collision():
        pygame.draw.rect(screen, YELLOW, player)
        pygame.display.update()
        print("💥 Collision! You crashed.")
        pygame.time.delay(1000)
        break

    pygame.draw.rect(screen, GREEN, player)
    pygame.display.update()

# Result
if won:
    screen.fill(BLUE)
    font = pygame.font.SysFont(None, 50)
    text = font.render("🏁 You reached the office!", True, WHITE)
    screen.blit(text, (30, HEIGHT // 2 - 20))
    pygame.display.update()
    pygame.time.delay(2000)

pygame.quit()
