"""Provided drone dynamics and Pygame window."""
from dataclasses import dataclass


@dataclass
class Drone:
    altitude: float = 2.5
    velocity: float = 0.0

    def advance(self, seconds):
        self.altitude = max(0.5, min(5.0, self.altitude + self.velocity * seconds))


def show_game(replay, sfreq):
    import pygame

    pygame.display.init()
    pygame.font.init()
    screen = pygame.display.set_mode((880, 520))
    pygame.display.set_caption("EEG to Drone")
    font = pygame.font.SysFont("Arial", 22)
    title = pygame.font.SysFont("Arial", 32, bold=True)
    clock = pygame.time.Clock()
    ink, muted, mint = (226, 235, 245), (150, 174, 199), (85, 225, 184)
    playhead, paused, running = 0.0, False, True
    total = len(replay.samples)

    def text(value, x, y, color=ink, face=font):
        screen.blit(face.render(value, True, color), (x, y))

    try:
        while running:
            dt = clock.tick(60) / 1000
            for event in pygame.event.get():
                if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
                    running = False
                elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE:
                    paused = not paused
            if not paused:
                playhead = min(total, playhead + dt * sfreq)
                replay.advance(int(playhead))
            if playhead >= total:
                replay.drone.velocity = 0.0
                replay.smoothed = None

            screen.fill((15, 23, 38))
            text("EEG TO DRONE", 30, 20, ink, title)
            text("PhysioNet · S001 · Run 14 · Recorded replay", 30, 59, muted)
            pygame.draw.rect(screen, (23, 38, 56), (30, 110, 465, 340), border_radius=12)
            for altitude in range(6):
                y = 417 - altitude * 55
                pygame.draw.line(screen, (55, 74, 92), (75, y), (470, y))
                text(f"{altitude} m", 38, y - 12, muted)

            drone = replay.drone
            y = round(417 - drone.altitude * 55)
            pygame.draw.line(screen, mint, (209, y), (329, y), 4)
            pygame.draw.rect(screen, mint, (248, y - 10, 42, 21), border_radius=6)
            for x in (209, 329):
                pygame.draw.ellipse(screen, mint, (x - 24, y - 11, 48, 10), 2)
            text(f"Altitude {drone.altitude:.2f} m", 175, 462)
            text(f"Replay time {playhead / sfreq:.1f} / {total / sfreq:.1f} s", 525, 119)
            status = "Replay complete" if playhead >= total else (
                "Cue gate open" if replay.smoothed is not None else "Hovering between decisions")
            text(status, 525, 157, muted)
            p = replay.smoothed
            text(f"Smoothed P(hands) {p:.2f}" if p is not None else "Waiting for a task window", 525, 211)
            pygame.draw.rect(screen, (55, 74, 92), (525, 246, 315, 16), border_radius=5)
            if p is not None:
                pygame.draw.rect(screen, mint, (525, 246, round(315 * p), 16), border_radius=5)
            movement = "UP" if drone.velocity > 0 else "DOWN" if drone.velocity < 0 else "HOVER"
            text(movement, 525, 304, mint, title)
            text("Paused · Space to resume" if paused else "Space to pause · Esc to quit",
                 525, 462, muted)
            pygame.display.flip()
    finally:
        pygame.quit()
