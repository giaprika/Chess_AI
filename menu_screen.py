# menu_screen.py

import sys
import pygame
import pygame_menu

# Constants (dùng cho menu)
MENU_WIDTH, MENU_HEIGHT = 540, 360

class MenuScreen:
    def __init__(self, screen, start_pvp, start_pvc):
        self.screen = screen
        self.start_pvp = start_pvp
        self.start_pvc = start_pvc

        # Load background
        self.bg = pygame.transform.scale(pygame.image.load("./data/images/bg.png"), (MENU_WIDTH, MENU_HEIGHT))

        # Menu theme
        font = pygame_menu.font.FONT_8BIT
        my_theme = pygame_menu.Theme(
            background_color=(0, 0, 0, 50),
            widget_background_color=(0, 255, 0, 50),
            widget_font_color=(255, 102, 255),
            widget_margin=(0, 10),
            widget_padding=10,
            widget_font=font,
            widget_font_size=24,
            title_font_size=24
        )

        # Create menu
        self.menu = pygame_menu.Menu('Chess game - ', MENU_WIDTH, MENU_HEIGHT, theme=my_theme)
        self.menu.add.button("Play", self.start_pvp)
        self.menu.add.button("Play with AI", self.start_pvc)
        self.menu.add.button("Quit", sys.exit, 1)

    def main_loop(self):
        while True:
            events = pygame.event.get()
            for e in events:
                if e.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            self.menu.update(events)
            self.screen.blit(self.bg, (0, 0))
            self.menu.draw(self.screen)
            pygame.display.update()
