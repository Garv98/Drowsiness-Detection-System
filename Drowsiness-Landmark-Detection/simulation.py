import pygame
import sys
import requests  # Add this import

# Initialize pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
LANE_WIDTH = 120
LANE_COUNT = 5
CAR_WIDTH = 60
CAR_HEIGHT = 100
CAR_SPEED = 5

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (128, 128, 128)
YELLOW = (255, 255, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

class Car:
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.target_x = x
        self.moving = True
        self.switching = False
        self.switch_speed = 8
        
    def update(self):
        # Move forward if not stopped
        if self.moving and not self.switching:
            self.y -= CAR_SPEED
            
        # Handle lane switching animation
        if self.switching:
            if abs(self.x - self.target_x) > 2:
                if self.x < self.target_x:
                    self.x += self.switch_speed
                else:
                    self.x -= self.switch_speed
            else:
                self.x = self.target_x
                self.switching = False
                self.moving = False  # Stop after reaching side lane
                
        # Reset position if car goes off screen
        if self.y < -CAR_HEIGHT:
            self.y = SCREEN_HEIGHT + CAR_HEIGHT
            self.moving = True
            
    def switch_lane(self, lane_index):
        if not self.switching and lane_index < LANE_COUNT:
            lane_center = (SCREEN_WIDTH - (LANE_COUNT * LANE_WIDTH)) // 2 + lane_index * LANE_WIDTH + LANE_WIDTH // 2
            self.target_x = lane_center - CAR_WIDTH // 2
            self.switching = True
            
    def draw(self, screen):
        # Draw car body
        pygame.draw.rect(screen, RED, (self.x, self.y, CAR_WIDTH, CAR_HEIGHT))
        # Draw car windows
        pygame.draw.rect(screen, BLUE, (self.x + 10, self.y + 10, CAR_WIDTH - 20, 30))
        # Draw car wheels
        pygame.draw.circle(screen, BLACK, (self.x + 15, self.y + CAR_HEIGHT - 10), 8)
        pygame.draw.circle(screen, BLACK, (self.x + CAR_WIDTH - 15, self.y + CAR_HEIGHT - 10), 8)

def draw_lanes(screen):
    # Calculate lane positions
    total_lane_width = LANE_COUNT * LANE_WIDTH
    start_x = (SCREEN_WIDTH - total_lane_width) // 2
    
    # Draw lane lines
    for i in range(LANE_COUNT + 1):
        x = start_x + i * LANE_WIDTH
        pygame.draw.line(screen, WHITE, (x, 0), (x, SCREEN_HEIGHT), 3)
        
    # Draw dashed center lines
    for i in range(1, LANE_COUNT):
        x = start_x + i * LANE_WIDTH
        for y in range(0, SCREEN_HEIGHT, 40):
            pygame.draw.line(screen, YELLOW, (x, y), (x, y + 20), 2)

def main():
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Car Lane Switching Game")
    clock = pygame.time.Clock()
    
    # Create car in the middle (lane 2, 0-indexed)
    middle_lane = 2
    lane_center = (SCREEN_WIDTH - (LANE_COUNT * LANE_WIDTH)) // 2 + middle_lane * LANE_WIDTH + LANE_WIDTH // 2
    car = Car(lane_center - CAR_WIDTH // 2, SCREEN_HEIGHT - 150)
    
    # Game state
    current_lane = middle_lane
    font = pygame.font.Font(None, 36)
    
    BASE_URL = "http://127.0.0.1:8000"  # FastAPI server URL

    o_pressed = False  # Track if 'O' has been pressed since last restart

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_o and not o_pressed:
                    # Switch to side lanes (lane 0 or 4)
                    if current_lane == middle_lane:
                        car.switch_lane(0)
                        current_lane = 0
                    elif current_lane == 0:
                        car.switch_lane(4)
                        current_lane = 4
                    elif current_lane == 4:
                        car.switch_lane(middle_lane)
                        current_lane = middle_lane
                    o_pressed = True  # Mark 'O' as pressed
                elif event.key == pygame.K_r:
                    # Reset car to middle lane and start moving
                    car.x = lane_center - CAR_WIDTH // 2
                    car.target_x = car.x
                    car.y = SCREEN_HEIGHT - 150
                    car.moving = True
                    car.switching = False
                    current_lane = middle_lane
                    o_pressed = False  # Allow 'O' again after reset

        # --- Check drowsie status from server and simulate 'O' or 'R' key if needed ---
        try:
            response = requests.get(f"{BASE_URL}/get_drowsie", timeout=0.2)
            drowsie_status = response.json().get("drowsie", False)
            if drowsie_status and not o_pressed:
                # Simulate 'O' key logic
                if current_lane == middle_lane:
                    car.switch_lane(0)
                    current_lane = 0
                elif current_lane == 0:
                    car.switch_lane(4)
                    current_lane = 4
                elif current_lane == 4:
                    car.switch_lane(middle_lane)
                    current_lane = middle_lane
                o_pressed = True  # Mark 'O' as pressed
            elif not drowsie_status:
                # Simulate 'R' key logic (restart)
                car.x = lane_center - CAR_WIDTH // 2
                car.target_x = car.x
                car.y = SCREEN_HEIGHT - 150
                car.moving = True
                car.switching = False
                current_lane = middle_lane
                o_pressed = False  # Allow 'O' again after reset
        except Exception as e:
            pass  # Ignore connection errors

        # Update
        car.update()
        
        # Draw
        screen.fill(GRAY)
        draw_lanes(screen)
        car.draw(screen)
        
        # Draw instructions
        instructions = [
            "Press 'O' to switch lanes (only once per game)",
            "Press 'R' to reset",
            f"Current lane: {current_lane + 1}",
            "Car stops when reaching side lanes"
        ]
        
        for i, instruction in enumerate(instructions):
            text = font.render(instruction, True, WHITE)
            screen.blit(text, (10, 10 + i * 40))
        
        pygame.display.flip()
        clock.tick(60)
    
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()