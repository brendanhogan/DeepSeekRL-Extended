from PIL import Image, ImageDraw, ImageFont, ImageFilter
import random
import json
import math
import os
import io
from typing import Tuple, List, Dict, Union, Optional

# Increased dimensions for clearer UI presentation
IMAGE_WIDTH = 384
IMAGE_HEIGHT = 384

# Internal high-resolution rendering dimensions - 2x oversampling
INTERNAL_RENDER_WIDTH = 768
INTERNAL_RENDER_HEIGHT = 768

# Aesthetic & Sizing Adjustments for macOS style
DEFAULT_FONT_SIZE = 14  # Slightly larger font size for better readability
TITLE_BAR_HEIGHT = 28  # Taller title bar for Safari style
WINDOW_MARGIN = 10
MIN_WINDOW_SCALE_FACTOR = 0.6  # Window should be larger for realism
INTERNAL_BUTTON_WIDTH = 55
INTERNAL_BUTTON_HEIGHT = 22
INTERNAL_BUTTON_CORNER_RADIUS = 6
INTERNAL_BUTTON_MARGIN = 8
MIN_SHAPE_SIZE = 15
MAX_SHAPE_SIZE = 35
SHAPE_MARGIN = 5

# Safari window styling
SAFARI_ADDRESS_BAR_HEIGHT = 32
SAFARI_TAB_HEIGHT = 28
SAFARI_TOOLBAR_HEIGHT = 26

# Dock specifications
DOCK_HEIGHT = 42
DOCK_ICON_SIZE = 36
DOCK_PADDING = 4
DOCK_BLUR_RADIUS = 10
DOCK_TRANSPARENCY = 0.85

# macOS Monterey Colors
WINDOW_BG_COLOR = "#FFFFFF"  # White for Safari background
WINDOW_TITLE_BAR_COLOR = "#F5F5F7"  # Light gray for title bar
SAFARI_ADDRESS_BAR_COLOR = "#F5F5F7"  # Light gray for address bar
BUTTON_START_COLOR = "#0D6EFD"  # Blue for primary buttons
BUTTON_STOP_COLOR = "#DC3545"  # Red for stop/cancel buttons
TEXT_COLOR = "#000000"  # Black text
DOCK_BG_COLOR = (200, 200, 200, 180)  # Semi-transparent light gray
CONTROL_RED = "#FF5F57"
CONTROL_YELLOW = "#FFBD2E"
CONTROL_GREEN = "#27C93F"
CONTROL_RED_OUTLINE = "#E0443E"
CONTROL_YELLOW_OUTLINE = "#DEA123"
CONTROL_GREEN_OUTLINE = "#1AAB29"
SHAPE_COLORS = ["#FFB3BA", "#FFDFBA", "#FFFFBA", "#BAFFC9", "#BAE1FF", "#E0BBE4"]

# Dock app icons colors (simplified for small size)
DOCK_ICONS = [
    {"name": "finder", "color": "#1E88E5"},       # Blue
    {"name": "messages", "color": "#43A047"},     # Green
    {"name": "safari", "color": "#29B6F6"},       # Light blue
    {"name": "photos", "color": "#F06292"},       # Pink
    {"name": "app_store", "color": "#42A5F5"},    # Light blue
    {"name": "terminal", "color": "#212121"},     # Black
    {"name": "spotify", "color": "#1DB954"}       # Green
]

class GUIElement:
    """Represents a single element in the GUI scene."""
    def __init__(self, name: str, center_x: int, center_y: int, bounding_box: tuple[int, int, int, int]):
        self.name = name
        self.center_x = center_x
        self.center_y = center_y
        self.bounding_box = bounding_box # (x_min, y_min, x_max, y_max)

    def to_dict(self):
        """Converts the element to a dictionary for JSON serialization."""
        return {
            "name": self.name,
            "center_x": self.center_x,
            "center_y": self.center_y,
            "bounding_box": self.bounding_box
        }

class GUIGenerator:
    """Generates synthetic GUI scenes with various elements."""

    def __init__(self, width=IMAGE_WIDTH, height=IMAGE_HEIGHT, seed=None):
        self.width = width
        self.height = height
        self.internal_width = INTERNAL_RENDER_WIDTH
        self.internal_height = INTERNAL_RENDER_HEIGHT
        self.elements: list[GUIElement] = []
        if seed is not None:
            random.seed(seed)
        
        # Improved font loading - try common sans-serif fonts
        font_names = ["SF-Pro-Display-Regular.otf", "DejaVuSans.ttf", "Helvetica.ttf", "Arial.ttf", "arial.ttf"]
        font_loaded = False
        for font_name in font_names:
            try:
                self.font = ImageFont.truetype(font_name, DEFAULT_FONT_SIZE)
                font_loaded = True
                break
            except IOError:
                continue
        if not font_loaded:
            try:
                self.font = ImageFont.load_default(size=DEFAULT_FONT_SIZE)
            except TypeError:
                self.font = ImageFont.load_default()

    def _add_element(self, name: str, bbox: tuple[int, int, int, int]):
        """Helper to create and add a GUIElement."""
        center_x = (bbox[0] + bbox[2]) // 2
        center_y = (bbox[1] + bbox[3]) // 2
        self.elements.append(GUIElement(name, center_x, center_y, bbox))

    def _create_rounded_rectangle_mask(self, size, radius):
        """Create a rounded rectangle mask."""
        mask = Image.new('L', size, 0)
        draw = ImageDraw.Draw(mask)
        draw.rounded_rectangle([(0, 0), (size[0], size[1])], radius=radius, fill=255)
        return mask

    def _create_safari_window(self, draw: ImageDraw.ImageDraw) -> tuple[int, int, int, int] | None:
        """Draws a Safari-style window. Returns window content bounding box."""
        min_w = int(self.width * MIN_WINDOW_SCALE_FACTOR)
        min_h = int(self.height * MIN_WINDOW_SCALE_FACTOR)

        # Ensure we leave enough space for the dock and margins
        max_height = self.height - 2 * WINDOW_MARGIN - DOCK_HEIGHT
        
        # Calculate minimum height needed for Safari components
        min_safari_height = TITLE_BAR_HEIGHT + SAFARI_TAB_HEIGHT + SAFARI_ADDRESS_BAR_HEIGHT + SAFARI_TOOLBAR_HEIGHT + 50
        
        # If minimum required height is greater than available space, reduce it to fit
        if min_safari_height > max_height:
            min_safari_height = max(min_h, max_height - 10)  # Leave at least 10px padding
        
        # Ensure w_height has a valid range
        if max_height <= min_safari_height:
            w_height = max_height  # Just use all available height
        else:
            w_height = random.randint(min_safari_height, max_height)
        
        # Calculate maximum width
        max_width = self.width - 2 * WINDOW_MARGIN
        w_width = random.randint(min_w, max_width)
        
        # Ensure we have valid vertical position range
        available_y_space = self.height - w_height - WINDOW_MARGIN - DOCK_HEIGHT
        if WINDOW_MARGIN >= available_y_space:
            w_y1 = WINDOW_MARGIN  # Just place at top margin if no room to vary
        else:
            w_y1 = random.randint(WINDOW_MARGIN, available_y_space)
            
        # Set horizontal position
        available_x_space = self.width - w_width - WINDOW_MARGIN
        if WINDOW_MARGIN >= available_x_space:
            w_x1 = WINDOW_MARGIN  # Center if no space
        else:
            w_x1 = random.randint(WINDOW_MARGIN, available_x_space)
            
        w_x2 = w_x1 + w_width
        w_y2 = w_y1 + w_height
        window_bbox = (w_x1, w_y1, w_x2, w_y2)

        # Draw window with slight shadow effect
        if hasattr(draw, "rounded_rectangle"):
            # Main window with shadow
            shadow_offset = 4
            shadow_bbox = (w_x1, w_y1, w_x2 + shadow_offset, w_y2 + shadow_offset)
            draw.rounded_rectangle(shadow_bbox, radius=6, fill=(0, 0, 0, 50))
            # Main window
            draw.rounded_rectangle(window_bbox, radius=6, fill="#FFFFFF", outline=None)
        else:
            # Fallback for older Pillow versions
            draw.rectangle(window_bbox, fill="#FFFFFF", outline=None)
        
        self._add_element("safari_window", window_bbox)

        # Draw title bar
        title_bar_bbox = (w_x1, w_y1, w_x2, w_y1 + TITLE_BAR_HEIGHT)
        draw.rectangle(title_bar_bbox, fill=WINDOW_TITLE_BAR_COLOR)

        # Window controls (traffic lights)
        control_y_center = w_y1 + TITLE_BAR_HEIGHT // 2
        
        # Red close button
        red_x_center = w_x1 + 14
        red_radius = 6
        red_bbox = (red_x_center - red_radius, control_y_center - red_radius, 
                    red_x_center + red_radius, control_y_center + red_radius)
        draw.ellipse(red_bbox, fill=CONTROL_RED, outline=CONTROL_RED_OUTLINE)
        self._add_element("window_close_button", red_bbox)

        # Yellow minimize button
        yellow_x_center = red_x_center + red_radius*2 + 8
        yellow_bbox = (yellow_x_center - red_radius, control_y_center - red_radius, 
                       yellow_x_center + red_radius, control_y_center + red_radius)
        draw.ellipse(yellow_bbox, fill=CONTROL_YELLOW, outline=CONTROL_YELLOW_OUTLINE)
        self._add_element("window_minimize_button", yellow_bbox)
        
        # Green maximize button
        green_x_center = yellow_x_center + red_radius*2 + 8
        green_bbox = (green_x_center - red_radius, control_y_center - red_radius, 
                      green_x_center + red_radius, control_y_center + red_radius)
        draw.ellipse(green_bbox, fill=CONTROL_GREEN, outline=CONTROL_GREEN_OUTLINE)
        self._add_element("window_maximize_button", green_bbox)
        
        # Draw Safari Window Title
        title_text = "Safari"
        try:
            title_bbox = draw.textbbox((0, 0), title_text, font=self.font)
            title_width = title_bbox[2] - title_bbox[0]
        except AttributeError:
            title_width = len(title_text) * (DEFAULT_FONT_SIZE * 0.6)
        
        title_x = w_x1 + (w_width - title_width) / 2
        title_y = w_y1 + (TITLE_BAR_HEIGHT - DEFAULT_FONT_SIZE) / 2 - 1
        draw.text((title_x, title_y), title_text, fill=TEXT_COLOR, font=self.font)

        # Draw Safari Tab Bar
        tab_bar_y1 = w_y1 + TITLE_BAR_HEIGHT
        tab_bar_y2 = tab_bar_y1 + SAFARI_TAB_HEIGHT
        tab_bar_bbox = (w_x1, tab_bar_y1, w_x2, tab_bar_y2)
        draw.rectangle(tab_bar_bbox, fill="#E0E0E2")

        # Draw a single active tab
        tab_padding = 12
        tab_text = "Apple"
        try:
            tab_text_bbox = draw.textbbox((0, 0), tab_text, font=self.font)
            tab_text_width = tab_text_bbox[2] - tab_text_bbox[0]
        except AttributeError:
            tab_text_width = len(tab_text) * (DEFAULT_FONT_SIZE * 0.6)
        
        tab_width = tab_text_width + 2 * tab_padding
        tab_x1 = w_x1 + 10
        tab_x2 = tab_x1 + tab_width
        
        # Active tab with lighter background
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle((tab_x1, tab_bar_y1 + 3, tab_x2, tab_bar_y2 - 1), 
                                  radius=4, fill="#FFFFFF")
        else:
            draw.rectangle((tab_x1, tab_bar_y1 + 3, tab_x2, tab_bar_y2 - 1), fill="#FFFFFF")
        
        tab_text_x = tab_x1 + tab_padding
        tab_text_y = tab_bar_y1 + (SAFARI_TAB_HEIGHT - DEFAULT_FONT_SIZE) / 2
        draw.text((tab_text_x, tab_text_y), tab_text, fill=TEXT_COLOR, font=self.font)

        # Draw Safari Address Bar
        address_bar_y1 = tab_bar_y2
        address_bar_y2 = address_bar_y1 + SAFARI_ADDRESS_BAR_HEIGHT
        address_bar_bbox = (w_x1, address_bar_y1, w_x2, address_bar_y2)
        draw.rectangle(address_bar_bbox, fill=SAFARI_ADDRESS_BAR_COLOR)
        
        # Draw URL field
        url_field_padding = 10
        url_field_height = 24
        url_field_y1 = address_bar_y1 + (SAFARI_ADDRESS_BAR_HEIGHT - url_field_height) / 2
        url_field_y2 = url_field_y1 + url_field_height
        url_field_x1 = w_x1 + 70  # Leave space for back/forward buttons
        url_field_x2 = w_x2 - 70  # Leave space for other controls
        
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle((url_field_x1, url_field_y1, url_field_x2, url_field_y2), 
                                 radius=4, fill="#FFFFFF")
        else:
            draw.rectangle((url_field_x1, url_field_y1, url_field_x2, url_field_y2), fill="#FFFFFF")
        
        # Draw URL text
        url_text = "apple.com"
        url_text_x = url_field_x1 + url_field_padding
        url_text_y = url_field_y1 + (url_field_height - DEFAULT_FONT_SIZE) / 2
        draw.text((url_text_x, url_text_y), url_text, fill="#777777", font=self.font)
        
        # Draw Safari Toolbar
        toolbar_y1 = address_bar_y2
        toolbar_y2 = toolbar_y1 + SAFARI_TOOLBAR_HEIGHT
        toolbar_bbox = (w_x1, toolbar_y1, w_x2, toolbar_y2)
        draw.rectangle(toolbar_bbox, fill="#F5F5F7")
        
        # Content area starts below all toolbars
        content_y1 = toolbar_y2
        content_area_bbox = (w_x1, content_y1, w_x2, w_y2)
        draw.rectangle(content_area_bbox, fill=WINDOW_BG_COLOR)
        
        return content_area_bbox

    def _draw_dock(self, image: Image.Image) -> None:
        """
        Draws the macOS dock using the static dock.png image and creates
        interactive elements for each specific app in the dock.
        """
        try:
            # Load the dock image
            dock_img = Image.open("dock.png")
            
            # Calculate dock dimensions and position
            dock_height = min(DOCK_HEIGHT * 2, self.height // 8)  # Make dock height proportional
            dock_aspect = dock_img.width / dock_img.height
            dock_width = int(dock_height * dock_aspect)
            
            # Resize the dock to fit properly
            dock_img = dock_img.resize((dock_width, dock_height), Image.LANCZOS if hasattr(Image, 'LANCZOS') else Image.ANTIALIAS)
            
            # Center the dock horizontally
            dock_x1 = (self.width - dock_width) // 2
            dock_y1 = self.height - dock_height - WINDOW_MARGIN
            
            # If the dock image has transparency (alpha channel), use it
            if dock_img.mode == 'RGBA':
                image.paste(dock_img, (dock_x1, dock_y1), dock_img)
            else:
                # Convert to RGBA if needed
                dock_img = dock_img.convert('RGBA')
                image.paste(dock_img, (dock_x1, dock_y1), dock_img)
            
            # Define the exact 10 apps in the dock from left to right
            dock_apps = [
                {"name": "finder", "label": "Finder"},
                {"name": "messages", "label": "Messages"},
                {"name": "safari", "label": "Safari"},
                {"name": "photos", "label": "Photos"},
                {"name": "iphone", "label": "iPhone"},
                {"name": "cursor", "label": "Cursor"},
                {"name": "terminal", "label": "Terminal"},
                {"name": "spotify", "label": "Spotify"},
                {"name": "preview", "label": "Preview"},
                {"name": "trash", "label": "Trash"}
            ]
            
            # Calculate precise positions for each app based on the static dock image
            # The dock has 10 apps, so divide the width evenly accounting for spacing
            num_apps = len(dock_apps)
            
            # First and last app positions (with margins)
            first_app_margin = dock_width * 0.02  # 2% margin from left
            last_app_margin = dock_width * 0.02   # 2% margin from right
            usable_width = dock_width - first_app_margin - last_app_margin
            
            # Calculate icon size (slightly smaller than dock height)
            icon_size = int(dock_height * 0.8)
            
            # Calculate spacing between icons
            icon_spacing = (usable_width - (icon_size * num_apps)) / (num_apps - 1)
            
            # Create interactive elements for each app
            for i, app in enumerate(dock_apps):
                # Calculate icon position
                if i == num_apps - 1:  # Trash (last icon) has special position
                    # Position Trash at the right end of the dock
                    icon_x = dock_x1 + dock_width - last_app_margin - icon_size
                else:
                    # Calculate position based on index
                    icon_x = dock_x1 + first_app_margin + i * (icon_size + icon_spacing)
                
                # Center icons vertically in the dock
                icon_y = dock_y1 + (dock_height - icon_size) // 2
                
                # Create bounding box for this app
                icon_bbox = (icon_x, icon_y, icon_x + icon_size, icon_y + icon_size)
                
                # Add element for this app icon
                self._add_element(f"dock_{app['name']}_icon", icon_bbox)
                
        except Exception as e:
            print(f"Error loading dock image: {e}, falling back to generated dock")
            # Fall back to the original dock drawing method if there's an error
            self._draw_generated_dock(image)

    def _draw_generated_dock(self, image: Image.Image) -> None:
        """Original method to draw a programmatically generated dock (fallback)."""
        # Calculate dock dimensions
        dock_width = min(self.width - 2*WINDOW_MARGIN, DOCK_ICON_SIZE * (len(DOCK_ICONS) + 2) + DOCK_PADDING * 2)
        dock_x1 = (self.width - dock_width) // 2
        dock_y1 = self.height - DOCK_HEIGHT - WINDOW_MARGIN
        dock_x2 = dock_x1 + dock_width
        dock_y2 = dock_y1 + DOCK_HEIGHT
        
        # Create a separate image for the dock with alpha channel
        dock_img = Image.new('RGBA', (dock_width, DOCK_HEIGHT), (0, 0, 0, 0))
        dock_draw = ImageDraw.Draw(dock_img)
        
        # Draw rounded dock background
        if hasattr(dock_draw, "rounded_rectangle"):
            dock_draw.rounded_rectangle([(0, 0), (dock_width, DOCK_HEIGHT)], 
                                        radius=DOCK_HEIGHT//2, 
                                        fill=DOCK_BG_COLOR)
        else:
            # Fallback for older Pillow
            dock_draw.rectangle([(0, 0), (dock_width, DOCK_HEIGHT)], fill=DOCK_BG_COLOR)
        
        # Apply blur if possible
        try:
            dock_img = dock_img.filter(ImageFilter.GaussianBlur(DOCK_BLUR_RADIUS))
        except:
            # Skip blur if not available
            pass
        
        # Draw icon placeholders
        icon_x = DOCK_PADDING + (dock_width - (len(DOCK_ICONS) * (DOCK_ICON_SIZE + DOCK_PADDING))) // 2
        icon_y = (DOCK_HEIGHT - DOCK_ICON_SIZE) // 2
        
        for i, icon_info in enumerate(DOCK_ICONS):
            # Create a square for the app icon
            icon_x1 = icon_x + i * (DOCK_ICON_SIZE + DOCK_PADDING)
            icon_y1 = icon_y
            icon_x2 = icon_x1 + DOCK_ICON_SIZE
            icon_y2 = icon_y1 + DOCK_ICON_SIZE
            
            # Draw app icon with rounded corners
            icon_img = Image.new('RGBA', (DOCK_ICON_SIZE, DOCK_ICON_SIZE), (0, 0, 0, 0))
            icon_draw = ImageDraw.Draw(icon_img)
            
            if hasattr(icon_draw, "rounded_rectangle"):
                icon_draw.rounded_rectangle(
                    [(0, 0), (DOCK_ICON_SIZE, DOCK_ICON_SIZE)], 
                    radius=DOCK_ICON_SIZE//5, 
                    fill=icon_info["color"]
                )
            else:
                # Fallback
                icon_draw.rectangle([(0, 0), (DOCK_ICON_SIZE, DOCK_ICON_SIZE)], fill=icon_info["color"])
            
            # Paste the icon onto the dock
            dock_img.paste(icon_img, (icon_x1, icon_y1), icon_img)
            
            # Add the icon as an element
            global_icon_bbox = (
                dock_x1 + icon_x1, 
                dock_y1 + icon_y1, 
                dock_x1 + icon_x2, 
                dock_y1 + icon_y2
            )
            self._add_element(f"dock_{icon_info['name']}_icon", global_icon_bbox)
            
        # Add a divider line
        divider_x = icon_x + len(DOCK_ICONS) * (DOCK_ICON_SIZE + DOCK_PADDING) + DOCK_PADDING//2
        dock_draw.line(
            [(divider_x, DOCK_HEIGHT//4), (divider_x, DOCK_HEIGHT*3//4)], 
            fill=(255, 255, 255, 180), 
            width=1
        )
        
        # Add trash icon
        trash_x = divider_x + DOCK_PADDING
        trash_y = icon_y
        
        trash_img = Image.new('RGBA', (DOCK_ICON_SIZE, DOCK_ICON_SIZE), (0, 0, 0, 0))
        trash_draw = ImageDraw.Draw(trash_img)
        
        if hasattr(trash_draw, "rounded_rectangle"):
            trash_draw.rounded_rectangle(
                [(0, 0), (DOCK_ICON_SIZE, DOCK_ICON_SIZE)], 
                radius=DOCK_ICON_SIZE//5, 
                fill="#AAAAAA"
            )
        else:
            trash_draw.rectangle([(0, 0), (DOCK_ICON_SIZE, DOCK_ICON_SIZE)], fill="#AAAAAA")
        
        # Paste the trash icon
        dock_img.paste(trash_img, (trash_x, trash_y), trash_img)
        
        # Add the trash icon as an element
        global_trash_bbox = (
            dock_x1 + trash_x, 
            dock_y1 + trash_y, 
            dock_x1 + trash_x + DOCK_ICON_SIZE, 
            dock_y1 + trash_y + DOCK_ICON_SIZE
        )
        self._add_element("dock_trash_icon", global_trash_bbox)
        
        # Paste the dock onto the main image
        image.paste(dock_img, (dock_x1, dock_y1), dock_img)
        
        # Add the whole dock as an element
        dock_bbox = (dock_x1, dock_y1, dock_x2, dock_y2)
        self._add_element("dock", dock_bbox)

    def _draw_menubar(self, draw: ImageDraw.ImageDraw) -> None:
        """
        Draws a macOS-style menubar at the top of the screen with interactive menu items.
        Each menu item is a fully interactive element that can be targeted and clicked.
        """
        # Draw a semi-transparent menubar
        menubar_height = 22
        menubar_bbox = (0, 0, self.width, menubar_height)
        
        # Draw translucent background
        draw.rectangle(menubar_bbox, fill=(255, 255, 255, 220))
        
        # Draw Apple logo
        apple_x = 10
        apple_y = menubar_height // 2
        apple_radius = 7
        draw.ellipse((apple_x - apple_radius, apple_y - apple_radius, 
                      apple_x + apple_radius, apple_y + apple_radius), 
                     fill="#000000")
        
        # Add the Apple logo as a clickable element
        apple_bbox = (apple_x - apple_radius, apple_y - apple_radius, 
                      apple_x + apple_radius, apple_y + apple_radius)
        self._add_element("menu_apple", apple_bbox)
        
        # Define all menu items with clear spacing
        menu_items = [
            {"name": "File", "id": "file"},
            {"name": "Edit", "id": "edit"},
            {"name": "View", "id": "view"},
            {"name": "History", "id": "history"},
            {"name": "Bookmarks", "id": "bookmarks"},
            {"name": "Window", "id": "window"},
            {"name": "Help", "id": "help"}
        ]
        
        menu_x = 30
        menu_y = (menubar_height - DEFAULT_FONT_SIZE) // 2
        menu_spacing = 16  # Space between menu items
        
        # Draw each menu item with proper spacing and add as interactive element
        for item in menu_items:
            # Calculate text dimensions
            try:
                text_bbox = draw.textbbox((0, 0), item["name"], font=self.font)
                text_width = text_bbox[2] - text_bbox[0]
                text_height = text_bbox[3] - text_bbox[1]
            except AttributeError:
                text_width = len(item["name"]) * (DEFAULT_FONT_SIZE * 0.6)
                text_height = DEFAULT_FONT_SIZE
            
            # Draw the text
            draw.text((menu_x, menu_y), item["name"], fill="#000000", font=self.font)
            
            # Create a slightly larger bounding box for better clickability
            item_bbox = (
                menu_x - 2,                  # Slightly wider left
                0,                           # Top of menubar
                menu_x + text_width + 2,     # Slightly wider right
                menubar_height               # Bottom of menubar
            )
            
            # Add menu item as interactive element with the same structure as buttons
            menu_item_name = f"menu_{item['id']}"
            self._add_element(menu_item_name, item_bbox)
            
            # Move to the next menu item position
            menu_x += text_width + menu_spacing
        
        # Draw right side elements (status icons)
        right_x = self.width - 10
        
        # Battery icon
        battery_width = 22
        battery_height = 12
        battery_x1 = right_x - battery_width
        battery_y1 = (menubar_height - battery_height) // 2
        battery_x2 = right_x
        battery_y2 = battery_y1 + battery_height
        
        # Draw a simple battery icon
        if hasattr(draw, "rounded_rectangle"):
            draw.rounded_rectangle(
                (battery_x1, battery_y1, battery_x2, battery_y2),
                radius=2, fill=None, outline="#000000"
            )
        else:
            draw.rectangle((battery_x1, battery_y1, battery_x2, battery_y2), 
                          outline="#000000")
        
        # Add battery as clickable item
        battery_bbox = (battery_x1, battery_y1, battery_x2, battery_y2)
        self._add_element("menu_battery", battery_bbox)
        
        right_x = battery_x1 - 10
        
        # WiFi icon
        wifi_radius = 8
        wifi_x = right_x - wifi_radius
        wifi_y = menubar_height // 2
        
        # Draw a simplified wifi icon
        for i in range(3):
            arc_radius = (i + 1) * 3
            draw.arc(
                (wifi_x - arc_radius, wifi_y - arc_radius, wifi_x + arc_radius, wifi_y + arc_radius),
                140, 400, fill="#000000", width=1
            )
        
        # Add WiFi as clickable item
        wifi_bbox = (wifi_x - wifi_radius - 2, 0, 
                     wifi_x + wifi_radius + 2, menubar_height)
        self._add_element("menu_wifi", wifi_bbox)
        
        # Add menubar as element (entire bar)
        self._add_element("menubar", menubar_bbox)

    def _draw_internal_buttons(self, draw: ImageDraw.ImageDraw, content_bbox: tuple[int, int, int, int]):
        """Draws 'start' and 'stop' buttons within the window's content area."""
        con_x1, con_y1, con_x2, con_y2 = content_bbox

        min_content_width = INTERNAL_BUTTON_WIDTH + 2 * INTERNAL_BUTTON_MARGIN
        min_content_height = INTERNAL_BUTTON_HEIGHT + 2 * INTERNAL_BUTTON_MARGIN
        if (con_x2 - con_x1) < min_content_width or \
           (con_y2 - con_y1) < min_content_height:
            return 

        def place_button(name: str, button_text_color: str, button_bg_color: str, existing_bboxes: list):
            max_attempts = 20
            for _ in range(max_attempts):
                # Ensure button fits
                btn_x1 = random.randint(con_x1 + INTERNAL_BUTTON_MARGIN, max(con_x1 + INTERNAL_BUTTON_MARGIN, con_x2 - INTERNAL_BUTTON_WIDTH - INTERNAL_BUTTON_MARGIN))
                btn_y1 = random.randint(con_y1 + INTERNAL_BUTTON_MARGIN, max(con_y1 + INTERNAL_BUTTON_MARGIN, con_y2 - INTERNAL_BUTTON_HEIGHT - INTERNAL_BUTTON_MARGIN))
                btn_x2 = btn_x1 + INTERNAL_BUTTON_WIDTH
                btn_y2 = btn_y1 + INTERNAL_BUTTON_HEIGHT
                current_bbox = (btn_x1, btn_y1, btn_x2, btn_y2)

                # Check if button is fully within content_bbox
                if not (btn_x1 >= con_x1 and btn_y1 >= con_y1 and btn_x2 <= con_x2 and btn_y2 <= con_y2):
                    continue
                    
                is_overlapping = False
                for eb_x1, eb_y1, eb_x2, eb_y2 in existing_bboxes:
                    if not (btn_x2 < eb_x1 or btn_x1 > eb_x2 or btn_y2 < eb_y1 or btn_y1 > eb_y2):
                        is_overlapping = True
                        break
                
                if not is_overlapping:
                    # Try to use rounded_rectangle if available (Pillow 9.0.0+)
                    if hasattr(draw, "rounded_rectangle"):
                        draw.rounded_rectangle(current_bbox, radius=INTERNAL_BUTTON_CORNER_RADIUS, fill=button_bg_color, outline=None)
                    else:
                        draw.rectangle(current_bbox, fill=button_bg_color, outline=None)
                    
                    try:
                        text_bbox = draw.textbbox((0,0), name, font=self.font)
                        text_width = text_bbox[2] - text_bbox[0]
                        text_height = text_bbox[3] - text_bbox[1]
                    except AttributeError: 
                        text_width = int(len(name) * (self.font.size if hasattr(self.font, 'size') else DEFAULT_FONT_SIZE * 0.6))
                        text_height = int(self.font.size if hasattr(self.font, 'size') else DEFAULT_FONT_SIZE)

                    text_x = btn_x1 + (INTERNAL_BUTTON_WIDTH - text_width) / 2
                    text_y = btn_y1 + (INTERNAL_BUTTON_HEIGHT - text_height) / 2 -1 # Small adjustment for better vertical centering
                    draw.text((text_x, text_y), name, fill=button_text_color, font=self.font)
                    
                    # Use consistent semantic names
                    semantic_name = "start_button" if name.lower() == "start" else "stop_button"
                    self._add_element(semantic_name, current_bbox)
                    existing_bboxes.append(current_bbox)
                    return True
            return False

        placed_button_bboxes = []
        
        # Use macOS style button colors
        place_button("Start", "#FFFFFF", BUTTON_START_COLOR, placed_button_bboxes)
        
        # Check if there's enough space for a second button
        if (con_x2 - con_x1) >= (INTERNAL_BUTTON_WIDTH * 2 + INTERNAL_BUTTON_MARGIN * 3) or \
           (con_y2 - con_y1) >= (INTERNAL_BUTTON_HEIGHT * 2 + INTERNAL_BUTTON_MARGIN * 3): 
            place_button("Stop", "#FFFFFF", BUTTON_STOP_COLOR, placed_button_bboxes)

    def _draw_random_shapes(self, draw: ImageDraw.ImageDraw, num_shapes: int):
        """Draws a number of random simple shapes on the canvas."""
        shape_types = ["rectangle", "ellipse"]
        outline_color = "#BFBFBF"

        for i in range(num_shapes):
            shape_type = random.choice(shape_types)
            size_w = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            size_h = random.randint(MIN_SHAPE_SIZE, MAX_SHAPE_SIZE)
            
            x1 = random.randint(SHAPE_MARGIN, self.width - size_w - SHAPE_MARGIN)
            y1 = random.randint(SHAPE_MARGIN, self.height - size_h - SHAPE_MARGIN)
            x2 = x1 + size_w
            y2 = y1 + size_h
            
            bbox = (x1, y1, x2, y2)
            color = random.choice(SHAPE_COLORS)
            element_name = f"random_{shape_type}_{i+1}"

            if shape_type == "rectangle":
                if hasattr(draw, "rounded_rectangle"):
                    draw.rounded_rectangle(bbox, radius=4, fill=color, outline=outline_color)
                else:
                draw.rectangle(bbox, fill=color, outline=outline_color)
            elif shape_type == "ellipse":
                draw.ellipse(bbox, fill=color, outline=outline_color)
            
            self._add_element(element_name, bbox)

    def generate_scene(self) -> tuple[Image.Image, str]:
        """Generates a new realistic macOS GUI scene with Safari-style window."""
        self.elements = [] 
        
        # Store original requested dimensions
        orig_width = self.width
        orig_height = self.height
        
        # Temporarily set dimensions to high resolution for rendering
        self.width = self.internal_width
        self.height = self.internal_height
        
        try:
            # Try to load the background image
            background = Image.open("background_img.png")
            # Resize to the high-resolution internal dimensions
            background = background.resize((self.width, self.height))
        except Exception as e:
            print(f"Error loading background image: {e}")
            # Fallback to a gradient background
            background = Image.new("RGB", (self.width, self.height), "#EAEAEA")
            draw = ImageDraw.Draw(background)
            for y in range(self.height):
                color_value = int(180 + (y / self.height) * 50)
                draw.line([(0, y), (self.width, y)], fill=(color_value, color_value, color_value))
        
        # Convert to RGBA to support transparency
        if background.mode != 'RGBA':
            background = background.convert('RGBA')
            
        # Create a drawing context
        draw = ImageDraw.Draw(background)
        
        # Draw the menubar
        self._draw_menubar(draw)
        
        # Draw the Safari window
        content_bbox = self._create_safari_window(draw)
        
        # Draw the dock at the bottom
        self._draw_dock(background)
        
        if content_bbox:
            # Draw buttons within the window content area
            self._draw_internal_buttons(draw, content_bbox)
        
        # Scale factors for coordinate conversion
        scale_x = orig_width / self.width
        scale_y = orig_height / self.height
        
        # Scale all elements' coordinates
        for element in self.elements:
            x_min, y_min, x_max, y_max = element.bounding_box
            scaled_bbox = (
                int(x_min * scale_x),
                int(y_min * scale_y),
                int(x_max * scale_x),
                int(y_max * scale_y)
            )
            element.bounding_box = scaled_bbox
            element.center_x = int(element.center_x * scale_x)
            element.center_y = int(element.center_y * scale_y)
        
        # Now resize the high-res image down to the requested dimensions
        # Use BICUBIC for sharper downsampling results
        background = background.resize(
            (orig_width, orig_height), 
            Image.BICUBIC if hasattr(Image, 'BICUBIC') else Image.ANTIALIAS
        )
        
        # Restore original dimensions
        self.width = orig_width
        self.height = orig_height

        elements_json = json.dumps([element.to_dict() for element in self.elements], indent=2)
        return background, elements_json

    def generate_scene_with_target(self) -> tuple[Image.Image, str, dict | None]:
        """
        Generates a realistic macOS GUI scene, selects a random interactive target element,
        and returns the image, JSON of all elements, and the target element's details.
        """
        image, all_elements_json = self.generate_scene()

        # Limited subset of interactive elements (as requested)
        preferred_target_names = [
            # Window controls
            "window_close_button", "window_minimize_button", "window_maximize_button",
            # Buttons
            "start_button", "stop_button", 
            # Selected dock icons
            "dock_messages_icon", "dock_cursor_icon", "dock_spotify_icon",
            # Selected menu items
            "menu_file", "menu_bookmarks", "menu_help", "menu_wifi"
        ]
        
        # Find all elements that match our preferred targets
        candidate_targets = []
        for elem in self.elements:
            if elem.name in preferred_target_names:
                candidate_targets.append(elem.to_dict())
        
        # If none of our preferred targets exist, fall back to any interactive element
        if not candidate_targets:
            interactive_element_names = [
                # Window controls
                "window_close_button", "window_minimize_button", "window_maximize_button",
                # Buttons
                "start_button", "stop_button", 
                # Dock icons (exact mapping to dock.png)
                "dock_finder_icon", "dock_messages_icon", "dock_safari_icon", 
                "dock_photos_icon", "dock_iphone_icon", "dock_cursor_icon",
                "dock_terminal_icon", "dock_spotify_icon", "dock_preview_icon", "dock_trash_icon",
                # Menu items
                "menu_apple", "menu_file", "menu_edit", "menu_view", "menu_history", 
                "menu_bookmarks", "menu_window", "menu_help", "menu_battery", "menu_wifi"
            ]
            
            for elem in self.elements:
                if elem.name in interactive_element_names:
                candidate_targets.append(elem.to_dict())
        
        if not candidate_targets:
            return image, all_elements_json, None
            
        selected_target = random.choice(candidate_targets)
        return image, all_elements_json, selected_target

    @staticmethod
    def plot_predictions(image: Image.Image, predictions_data: str | list, pred_color="#FF00FF", truth_color="#00FF00", default_font_size=9, x_marker_size=4) -> Image.Image:
        """
        Plots predicted and/or ground truth annotations on an image.
        predictions_data: JSON string or list of dicts.
        Elements in predictions_data can have an optional 'is_truth': True key to be styled with truth_color.
        """
        output_image = image.copy()
        draw = ImageDraw.Draw(output_image)
        img_width, img_height = output_image.size # Get image dimensions
        
        try:
            if isinstance(predictions_data, str):
                predictions = json.loads(predictions_data)
            elif isinstance(predictions_data, list):
                predictions = predictions_data
            else:
                print(f"Warning: Invalid type for predictions_data: {type(predictions_data)}. Expected str or list.")
                predictions = []
        except json.JSONDecodeError:
            print(f"Error decoding predictions JSON: {predictions_data[:100]}...")
            predictions = []

        # Font for labels on plot
        plot_font = None
        font_names = ["DejaVuSans.ttf", "Helvetica.ttf", "Arial.ttf", "arial.ttf"]
        for font_name in font_names:
            try:
                plot_font = ImageFont.truetype(font_name, default_font_size)
                break
            except IOError:
                continue
        if not plot_font:
            try:
                plot_font = ImageFont.load_default(size=default_font_size)
            except TypeError:
                plot_font = ImageFont.load_default()


        for pred_elem in predictions:
            if not isinstance(pred_elem, dict): 
                print(f"Warning: Skipping non-dict element in predictions list: {pred_elem}")
                continue

            name = pred_elem.get("name", "unknown")
            bbox = pred_elem.get("bounding_box")
            center_x = pred_elem.get("center_x")
            center_y = pred_elem.get("center_y")

            current_color = truth_color if pred_elem.get("is_truth") else pred_color
            
            if bbox and isinstance(bbox, (list, tuple)) and len(bbox) == 4:
                pred_x1, pred_y1, pred_x2, pred_y2 = bbox
                draw.rectangle((pred_x1, pred_y1, pred_x2, pred_y2), outline=current_color, width=2) # Thicker outline for bbox
                
                try:
                    text_bbox_plot = plot_font.getbbox(name) # Pillow 9+
                    text_height = text_bbox_plot[3] - text_bbox_plot[1]
                except AttributeError: 
                    text_height = default_font_size 
                
                label_y = pred_y1 - text_height - 2 if pred_y1 - text_height -2 > 0 else pred_y1 + 2
                draw.text((pred_x1 + 2, label_y), name, fill=current_color, font=plot_font)

            elif center_x is not None and center_y is not None:
                # Clamp coordinates before drawing the 'X' marker
                plot_x = max(0, min(img_width - 1, int(center_x)))
                plot_y = max(0, min(img_height - 1, int(center_y)))
                
                # Draw an 'X' marker for point predictions at the (potentially clamped) coordinates
                draw.line([(plot_x - x_marker_size, plot_y - x_marker_size),
                           (plot_x + x_marker_size, plot_y + x_marker_size)],
                          fill=current_color, width=2)
                draw.line([(plot_x - x_marker_size, plot_y + x_marker_size),
                           (plot_x + x_marker_size, plot_y - x_marker_size)],
                          fill=current_color, width=2)
                
                try:
                    text_bbox_plot = plot_font.getbbox(name)
                    text_height = text_bbox_plot[3] - text_bbox_plot[1]
                except AttributeError: 
                    text_height = default_font_size
                
                label_y_offset = text_height + x_marker_size + 2
                label_y = plot_y - label_y_offset if plot_y - label_y_offset > 0 else plot_y + x_marker_size + 2
                draw.text((plot_x + x_marker_size + 2, label_y), name, fill=current_color, font=plot_font)
        return output_image

if __name__ == '__main__':

    # Original main block for generating polished examples and comparisons:
    generator_main_example = GUIGenerator(seed=42)
    generated_image_main, elements_json_data_main = generator_main_example.generate_scene()
    generated_image_main.save("gui_scene_polished_example.png")
    with open("gui_scene_polished_elements.json", "w") as f:
        f.write(elements_json_data_main)
    print("Generated: gui_scene_polished_example.png, gui_scene_polished_elements.json")

    gt_elements_list = json.loads(elements_json_data_main)
    for elem in gt_elements_list:
        elem["is_truth"] = True
    image_with_ground_truth = GUIGenerator.plot_predictions(generated_image_main, gt_elements_list)
    image_with_ground_truth.save("gui_scene_polished_with_ground_truth.png")
    print("Generated: gui_scene_polished_with_ground_truth.png")

    mock_predictions_list = []
    original_elements = json.loads(elements_json_data_main)
    if original_elements:
        for i, elem_data in enumerate(original_elements):
            pred_name = elem_data["name"]
            if "window_minimize_button" in pred_name: 
                continue
            if i % 2 == 0 and "random" not in pred_name: 
                mock_predictions_list.append({
                    "name": "pred_" + pred_name + "_click",
                    "center_x": elem_data["center_x"] + random.randint(-4, 4),
                    "center_y": elem_data["center_y"] + random.randint(-4, 4),
                    "is_truth": False
                })
            else: 
                new_bbox = [b + random.randint(-7, 7) for b in elem_data["bounding_box"]]
                mock_predictions_list.append({
                    "name": "pred_" + pred_name,
                    "center_x": (new_bbox[0] + new_bbox[2]) // 2,
                    "center_y": (new_bbox[1] + new_bbox[3]) // 2,
                    "bounding_box": new_bbox,
                    "is_truth": False
                })
        mock_predictions_list.append({
            "name": "false_positive_click",
            "center_x": random.randint(20, IMAGE_WIDTH - 20),
            "center_y": random.randint(20, IMAGE_HEIGHT - 20),
            "is_truth": False
        })
    image_with_mock_predictions = GUIGenerator.plot_predictions(generated_image_main, mock_predictions_list)
    image_with_mock_predictions.save("gui_scene_polished_with_mock_predictions.png")
    print("Generated: gui_scene_polished_with_mock_predictions.png")

    combined_plot_data = gt_elements_list + mock_predictions_list
    comparison_image_with_both = GUIGenerator.plot_predictions(generated_image_main, combined_plot_data)
    comparison_image_with_both.save("gui_scene_polished_comparison.png")
    print("Generated: gui_scene_polished_comparison.png (green=truth, magenta=prediction)")

    # --- Generate 15 examples for variability check ---
    examples_dir = "examples"
    os.makedirs(examples_dir, exist_ok=True)
    print(f"\nGenerating 15 examples in '{examples_dir}/' directory...")
    example_generator_loop = GUIGenerator() 
    for i in range(15):
        example_image, _ = example_generator_loop.generate_scene() # Using new realistic generate_scene
        example_image.save(os.path.join(examples_dir, f"gui_scene_example_{i+1:02d}.png"))
    print(f"Finished generating 15 examples in '{examples_dir}/'.") 

    # Generate with target
    print("\n--- Generating Example with Target ---")
    target_generator = GUIGenerator(seed=123)
    target_image, target_json, selected_target = target_generator.generate_scene_with_target()
    target_image.save("gui_scene_target_example.png")
    print("Saved: gui_scene_target_example.png")
    print("Target JSON sample:")
    # print(target_json)
    print("Selected target:")
    print(json.dumps(selected_target, indent=2) if selected_target else "None")

    # Plot target on image
    if selected_target:
         plot_data_target = [{**selected_target, "is_truth": True, "name": "TARGET_"+selected_target["name"]}]
         img_with_target = GUIGenerator.plot_predictions(target_image, plot_data_target, truth_color="cyan")
         img_with_target.save("gui_scene_WITH_TARGET.png")
         print("Saved: gui_scene_WITH_TARGET.png") 