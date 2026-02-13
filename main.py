import pygame
import sys
import random
import cv2
import mediapipe as mp
import numpy as np

# --- CONSTANTS ---
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 400
FPS = 60
GROUND_Y = SCREEN_HEIGHT - 50

# --- INITIALIZE ---
pygame.init()
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Gesture Dino - Manual + AI")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Arial", 20, bold=True)

# --- AI INITIALIZATION ---
AI_ENABLED = False
try:
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5
    )
    cap = cv2.VideoCapture(0)
    AI_ENABLED = True
except Exception as e:
    print(f"AI Initialization Failed: {e}. Falling back to Keyboard only.")

class Dino:
    def __init__(self):
        self.rect = pygame.Rect(50, GROUND_Y - 40, 40, 40)
        self.color = (50, 200, 50)
        self.vel_y = 0
        self.gravity = 1.0
        self.jump_strength = -18
        self.is_jumping = False

    def jump(self):
        if not self.is_jumping:
            self.vel_y = self.jump_strength
            self.is_jumping = True

    def update(self):
        self.vel_y += self.gravity
        self.rect.y += self.vel_y
        if self.rect.y > GROUND_Y - 40:
            self.rect.y = GROUND_Y - 40
            self.is_jumping = False

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

class Cactus:
    def __init__(self):
        self.rect = pygame.Rect(SCREEN_WIDTH, GROUND_Y - 30, 20, 30)
        self.color = (200, 50, 50)
        self.speed = 8

    def update(self):
        self.rect.x -= self.speed

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, self.rect)

dino = Dino()
obstacles = []
score = 0

def main():
    global score
    running = True
    
    while running:
        screen.fill((255, 255, 255))
        
        # 1. GESTURE PROCESSING
        if AI_ENABLED:
            ret, frame = cap.read()
            if ret:
                frame = cv2.flip(frame, 1) # Mirror effect
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_lms in results.multi_hand_landmarks:
                        # Index Finger Tip (8) vs Index Finger Knuckle (5)
                        # If tip is higher (lower Y) than knuckle, JUMP
                        if hand_lms.landmark[8].y < hand_lms.landmark[5].y:
                            dino.jump()
                
                # Show Camera Preview in Corner
                preview = cv2.resize(frame, (120, 90))
                preview = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
                preview = np.rot90(preview)
                cam_surf = pygame.surfarray.make_surface(preview)
                screen.blit(cam_surf, (SCREEN_WIDTH - 130, 10))

        # 2. EVENT HANDLING
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    dino.jump()

        # 3. GAME LOGIC
        if random.randint(0, 100) < 2:
            obstacles.append(Cactus())

        dino.update()
        for ob in obstacles[:]:
            ob.update()
            if ob.rect.right < 0:
                obstacles.remove(ob)
                score += 1
            if dino.rect.colliderect(ob.rect):
                print(f"Collision! Final Score: {score}")
                running = False

        # 4. DRAWING
        pygame.draw.line(screen, (0, 0, 0), (0, GROUND_Y), (SCREEN_WIDTH, GROUND_Y), 2)
        dino.draw(screen)
        for ob in obstacles:
            ob.draw(screen)
        
        status = "AI ACTIVE" if AI_ENABLED else "AI ERROR - Manual Mode"
        info_text = font.render(f"Score: {score} | {status}", True, (0, 0, 0))
        screen.blit(info_text, (20, 20))

        pygame.display.flip()
        clock.tick(FPS)

    if AI_ENABLED: cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()