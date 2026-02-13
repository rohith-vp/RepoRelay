import pygame
import cv2
import mediapipe as mp
import numpy as np
import sys
import random
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Setup
pygame.init()
WIDTH, HEIGHT = 800, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()
GROUND_Y = 350

# --- THE GESTURE ENGINE ---
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=1)
detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)

# --- THE GAME CLASSES ---
class Dino:
    def __init__(self):
        self.rect = pygame.Rect(50, GROUND_Y - 40, 40, 40)
        self.vel_y = 0
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.vel_y = -18
            self.is_jumping = True

    def update(self):
        if self.is_jumping:
            self.rect.y += self.vel_y
            self.vel_y += 1  # Gravity
            if self.rect.y >= GROUND_Y - 40:
                self.rect.y = GROUND_Y - 40
                self.is_jumping = False

class Cactus:
    def __init__(self):
        self.rect = pygame.Rect(WIDTH, GROUND_Y - 40, 25, 40)
        self.speed = 8

    def update(self):
        self.rect.x -= self.speed

# --- INITIALIZE OBJECTS ---
dino = Dino()
obstacles = []
score = 0
font = pygame.font.SysFont("Arial", 24, bold=True)

while True:
    screen.fill((255, 255, 255))
    
    # 1. Capture Camera & Detect Gesture
    success, frame = cap.read()
    if success:
        frame = cv2.flip(frame, 1)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        result = detector.detect(mp_image)
        
        if result.hand_landmarks:
            # Tip (8) vs PIP Joint (6)
            tip_y = result.hand_landmarks[0][8].y
            pip_y = result.hand_landmarks[0][6].y
            if tip_y < pip_y:
                dino.jump()

        # Optional: Display small camera preview
        preview = cv2.resize(frame, (120, 90))
        preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
        preview = np.rot90(preview)
        cam_surf = pygame.surfarray.make_surface(preview)
        screen.blit(cam_surf, (WIDTH - 130, 10))

    # 2. Pygame Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            cap.release()
            pygame.quit()
            sys.exit()

    # 3. Game Logic & Spawning
    dino.update()
    
    # Spawn cactuses
    if len(obstacles) == 0 or obstacles[-1].rect.x < WIDTH - random.randint(250, 450):
        obstacles.append(Cactus())

    for ob in obstacles[:]:
        ob.update()
        if ob.rect.right < 0:
            obstacles.remove(ob)
            score += 1
        
        # Collision
        if dino.rect.colliderect(ob.rect):
            print(f"GAME OVER! Score: {score}")
            cap.release()
            pygame.quit()
            sys.exit()

    # 4. Render
    pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (WIDTH, GROUND_Y), 2)
    pygame.draw.rect(screen, (0, 200, 0), dino.rect)
    for ob in obstacles:
        pygame.draw.rect(screen, (200, 0, 0), ob.rect)
    
    score_surf = font.render(f"Score: {score}", True, (0, 0, 0))
    screen.blit(score_surf, (20, 20))

    pygame.display.flip()
    clock.tick(60)