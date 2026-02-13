import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# --- Initialization ---
pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Hand-Track Dino")
clock = pygame.time.Clock()
GROUND_Y = 350

# --- Assets ---
try:
    dino_img = pygame.image.load('dino.png').convert_alpha()
    dino_img = pygame.transform.scale(dino_img, (44, 48))
    cactus_img = pygame.image.load('cactus.png').convert_alpha()
    cactus_img = pygame.transform.scale(cactus_img, (30, 50))
    # Note: Scaled background to screen size for better visuals
    background_img = pygame.image.load('backgroundd.jpg').convert()
    background_img = pygame.transform.scale(background_img, (WIDTH, HEIGHT))
    use_graphics = True
except:
    use_graphics = False
    print("Assets not found. Using shapes instead.")

# --- MediaPipe Setup ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(
    base_options=base_options, 
    num_hands=1,
    running_mode=vision.RunningMode.VIDEO
)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# --- Classes ---
class Dino:
    def __init__(self):
        self.rect = pygame.Rect(50, GROUND_Y - 48, 44, 48)
        self.vel_y = 0
        self.gravity = 1.2
        self.jump_power = -18
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.vel_y = self.jump_power
            self.is_jumping = True

    def update(self):
        if self.is_jumping:
            self.rect.y += self.vel_y
            self.vel_y += self.gravity
            if self.rect.y >= GROUND_Y - 48:
                self.rect.y = GROUND_Y - 48
                self.is_jumping = False
                self.vel_y = 0

    def draw(self):
        if use_graphics:
            screen.blit(dino_img, self.rect)
        else:
            pygame.draw.rect(screen, (46, 204, 113), self.rect, border_radius=5)

class Cactus:
    def __init__(self):
        self.width = 30
        self.height = 50
        self.rect = pygame.Rect(WIDTH, GROUND_Y - self.height, self.width, self.height)
        self.speed = 8

    def update(self):
        self.rect.x -= self.speed

    def draw(self):
        if use_graphics:
            screen.blit(cactus_img, self.rect)
        else:
            pygame.draw.rect(screen, (231, 76, 60), self.rect, border_radius=3)

# --- Game State ---
dino = Dino()
obstacles = []
score = 0
font = pygame.font.SysFont("Arial", 24, bold=True)

# --- Main Loop ---
running = True
while running:
    # 1. Background Logic (Fixed Indentation)
    if use_graphics:
        screen.blit(background_img, (0, 0))
    else:
        screen.fill((240, 240, 240))

    current_time_ms = pygame.time.get_ticks()
    
    # 2. Camera & Hand Tracking
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Landmark detection
        result = detector.detect_for_video(mp_image, current_time_ms)
        
        if result.hand_landmarks:
            # MediaPipe uses normalized coordinates (0.0 to 1.0)
            # Index Finger Tip (8) vs Index Finger PIP (6)
            tip_y = result.hand_landmarks[0][8].y
            pip_y = result.hand_landmarks[0][6].y
            
            # Gesture: If tip is significantly higher than PIP, jump
            if tip_y < pip_y - 0.05: 
                dino.jump()

        # Mini Camera Preview
        preview = cv2.resize(frame, (120, 90))
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        # Fix: Pygame uses (Width, Height) surface, NumPy uses (Height, Width, Colors)
        preview_rotated = np.transpose(preview_rgb, (1, 0, 2))
        cam_surf = pygame.surfarray.make_surface(preview_rotated)
        screen.blit(cam_surf, (WIDTH - 130, 10))

    # 3. Event Handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                dino.jump()

    # 4. Game Updates
    dino.update()
    
    # Obstacle Spawning
    if not obstacles or obstacles[-1].rect.x < WIDTH - random.randint(300, 600):
        obstacles.append(Cactus())

    for ob in obstacles[:]:
        ob.update()
        if ob.rect.right < 0:
            obstacles.remove(ob)
            score += 1
        
        if dino.rect.colliderect(ob.rect):
            print(f"Game Over! Score: {score}")
            running = False

    # 5. Drawing
    pygame.draw.rect(screen, (100, 100, 100), (0, GROUND_Y, WIDTH, 5)) # Floor
    dino.draw()
    for ob in obstacles:
        ob.draw()
    
    score_surf = font.render(f"SCORE: {score}", True, (50, 50, 50))
    screen.blit(score_surf, (20, 20))

    pygame.display.flip()
    clock.tick(60)

# Clean up
detector.close()
cap.release()
pygame.quit()
sys.exit()