import pygame
import pygame_gui
import json
import os
import random
import time


class EnvironmentSetupInterface:
    def __init__(self, screen, manager):
        self.screen = screen
        self.manager = manager
        self.clock = pygame.time.Clock()

        self.screen_width, self.screen_height = self.screen.get_size()
        self.environment_width = 200
        self.environment_height = 200
        self.grid_top_margin = 200  # Reduced from 220 to move grid up
        self.left_panel_width = 300
        self.right_panel_width = 320
        self.instruction_height = 14
        self.grid_bottom_margin = self.instruction_height + 38  # Increased from 18 to 38
        self.grid_size = min(
            self.screen_width - self.left_panel_width - self.right_panel_width,
            self.screen_height - self.grid_top_margin - self.grid_bottom_margin
        )
        self.cell_size = self.grid_size // max(self.environment_width, self.environment_height)

        self.hyperparameters = None
        self.hyperparameter_entries = {}

        self.growth_regions = []
        self.growth_region_data = []
        self.drawing = False
        self.start_pos = None
        self.current_region = None
        self.walls_top_bottom = False
        self.walls_left_right = False
        self.selected_region = None

        # Define colors for dark theme
        self.bg_color = (30, 30, 30)  # Dark gray
        self.grid_color = (60, 60, 60)  # Lighter gray for grid
        self.text_color = (200, 200, 200)  # Light gray for text
        self.growth_region_color = (0, 100, 0, 128)  # Dark green with transparency
        self.selected_region_border_color = (255, 0, 0)  # Red for the border
        self.selected_region_fill_color = (0, 100, 0, 128)  # Dark green with transparency
        self.current_region_color = (0, 0, 100, 128)  # Dark blue with transparency
        self.wall_color = (100, 100, 100)  # Medium gray for walls

        self.font = pygame.font.Font(None, 24)
        self.instruction_font = pygame.font.Font(None, 20)

        self.instructions = [
            "Left-click and drag to create a growth region",
            "Left-click to select a region",
            "Right-click to delete a region"
        ]

        self.click_start_pos = None
        self.click_threshold = 5  # pixels

        self.agents_can_move = False
        self.seed = "RANDOM"  # Default seed value

        self.invalid_fields = set()
        self.setup_ui()
        self.load_config()

    def setup_ui(self):
        # Update UI colors for dark theme with improved text visibility and highlighting
        self.manager.get_theme().load_theme({
            'defaults': {
                'colours': {
                    'normal_bg': '#3a3a3a',
                    'hovered_bg': '#4a4a4a',
                    'disabled_bg': '#2c2c2c',
                    'selected_bg': '#1c1c1c',
                    'active_bg': '#5a5a5a',
                    'normal_text': '#ffffff',
                    'hovered_text': '#ffffff',
                    'selected_text': '#ffffff',
                    'disabled_text': '#6d736f',
                    'link_text': '#0000EE',
                    'link_hover': '#2020FF',
                    'link_selected': '#551A8B',
                    'text_shadow': '#777777',
                    'normal_border': '#1c1c1c',
                    'hovered_border': '#c7c7c7',
                    'disabled_border': '#808080',
                    'selected_border': '#8080B0',
                    'active_border': '#8080B0',
                    'filled_bar': '#f4251b',
                    'unfilled_bar': '#CCCCCC',
                    'text_cursor': '#FFFFFF'
                }
            },
            'text_entry_line': {
                'colours': {
                    'dark_bg': '#2c2c2c',
                    'selected_bg': '#D3d3d3',
                    'normal_text': '#FFFFFF',
                    'invalid_text': '#FF0000',
                    'selected_text': '#000000',
                    'normal_border': '#4a4a4a',
                    'selected_border': '#8080B0',
                    'edit_text': '#FFFFFF',
                },
                'misc': {
                    'border_width': '1',
                    'shadow_width': '2',
                    'padding': '4,2',
                    'text_horiz_alignment': 'left',
                    'text_vert_alignment': 'center'
                }
            },
            'label': {
                'font': {
                    'size': '14',
                    'bold': '0',
                    'italic': '0',
                }
            },
            '#bold_label': {
                'font': {
                    'size': '14',
                    'bold': '1',
                    'italic': '0',
                }
            }
        })

        # Add left panel for hyperparameters
        self.left_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(0, 0, self.left_panel_width, self.screen_height),
            manager=self.manager
        )

        # Add title for left panel
        self.left_panel_title = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, 280, 30),
            text="Simulation Hyperparameters",
            manager=self.manager,
            container=self.left_panel,
            object_id="#bold_label"
        )

        # Hyperparameters
        # Define hyperparameters with their default values
        self.hyperparameters = [
            ("Consumption Rate", 0.5),
            ("Basal Metabolic Cost", 0.002),
            ("Neuron Cost", 0.0002),
            ("Synapse Cost", 0.00002),
            ("Replication Cost", 0.01),
            ("Replication Survivability Scalar", 6),
            ("Replication Cooldown", 6),
            ("Movement Cost", 0.05),
            ("Transfer Rate", 0.2),
            ("Crossover Probability", 0.2),
            ("Mutation Probability", 0.08),
            ("Compatibility Threshold", 0.16),
            ("Number of Communication Signals", 4),
            ("Progenitor Update Threshold", 500)
        ]

        self.hyperparameter_entries = {}

        label_height = 20
        entry_height = 25
        spacing = 10
        start_y = 50

        for i, (param_name, default_value) in enumerate(self.hyperparameters):
            y_pos = start_y + i * (label_height + entry_height + spacing)

            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(10, y_pos, 280, label_height),
                text=param_name + ":",
                manager=self.manager,
                container=self.left_panel
            )

            entry = pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect(10, y_pos + label_height, 280, entry_height),
                manager=self.manager,
                container=self.left_panel
            )
            if isinstance(default_value, float):
                formatted_value = f"{default_value:.10f}".rstrip('0').rstrip('.')
                entry.set_text(formatted_value)
            else:
                entry.set_text(str(default_value))

            self.hyperparameter_entries[param_name] = entry

        # Calculate the center of the screen, excluding the left and right panels
        center_x = (self.screen_width - self.left_panel_width - self.right_panel_width) / 2 + self.left_panel_width

        # Width and Height inputs
        label_width = 60
        input_width = 50
        input_height = 30
        input_spacing = 20
        button_width = 120

        # Calculate total width of all elements
        total_width = (label_width * 2) + (input_width * 2) + button_width + (input_spacing * 4)

        # Calculate start x position to center the group
        start_x = center_x - (total_width / 2)

        # Row 1: Save/Load buttons
        save_load_button_width = 150
        save_load_button_height = 25
        save_load_button_spacing = 10
        total_save_load_width = save_load_button_width * 2 + save_load_button_spacing

        self.save_config_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(center_x - total_save_load_width / 2, 10, save_load_button_width,
                                      save_load_button_height),
            text="Save Config",
            manager=self.manager
        )

        self.load_config_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(
                center_x - total_save_load_width / 2 + save_load_button_width + save_load_button_spacing, 10,
                save_load_button_width, save_load_button_height),
            text="Load Config",
            manager=self.manager
        )

        # Row 2: Width, Height, Apply Size
        row2_y = 45
        compact_input_height = 25
        compact_spacing = 8

        width_label_width = 45
        width_input_width = 45
        height_label_width = 50
        height_input_width = 45
        apply_button_width = 80

        total_row2_width = (width_label_width + width_input_width + height_label_width +
                           height_input_width + apply_button_width + compact_spacing * 4)
        row2_start_x = center_x - total_row2_width / 2

        self.width_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(row2_start_x, row2_y, width_label_width, compact_input_height),
            text="Width:",
            manager=self.manager
        )

        self.width_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(row2_start_x + width_label_width + compact_spacing, row2_y,
                                     width_input_width, compact_input_height),
            manager=self.manager
        )
        self.width_entry.set_text(str(self.environment_width))

        self.height_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(row2_start_x + width_label_width + width_input_width + compact_spacing * 2,
                                     row2_y, height_label_width, compact_input_height),
            text="Height:",
            manager=self.manager
        )

        self.height_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(row2_start_x + width_label_width + width_input_width + height_label_width + compact_spacing * 3,
                                     row2_y, height_input_width, compact_input_height),
            manager=self.manager
        )
        self.height_entry.set_text(str(self.environment_height))

        self.apply_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(row2_start_x + width_label_width + width_input_width + height_label_width + height_input_width + compact_spacing * 4,
                                     row2_y, apply_button_width, compact_input_height),
            text="Apply Size",
            manager=self.manager
        )

        # Row 3: Seed controls
        row3_y = 80
        seed_label_width = 40  # Increased from 35 to 40
        seed_input_width = 95  # Reduced from 100 to 95 to maintain total width
        seed_button_width = 70

        total_seed_width = seed_label_width + seed_input_width + seed_button_width + (compact_spacing * 2)
        seed_start_x = center_x - (total_seed_width / 2)

        self.seed_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(seed_start_x, row3_y, seed_label_width, compact_input_height),
            text="Seed:",
            manager=self.manager
        )

        self.seed_entry = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect(seed_start_x + seed_label_width + compact_spacing, row3_y,
                                     seed_input_width, compact_input_height),
            manager=self.manager
        )
        self.seed_entry.set_text(self.seed)

        self.random_seed_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(seed_start_x + seed_label_width + seed_input_width + compact_spacing * 2,
                                     row3_y, seed_button_width, compact_input_height),
            text="Random",
            manager=self.manager
        )

        button_y_offset = 105
        button_height = 30
        button_spacing = 10

        # Agents can move checkbox
        checkbox_height = 25
        checkbox_x = center_x
        checkbox_y = button_y_offset

        self.agents_move_checkbox = CustomCheckbox(
            checkbox_x, checkbox_y, checkbox_height,
            "Agents can move", self.manager, text_color=self.text_color
        )
        self.agents_move_checkbox.is_checked = self.agents_can_move

        # Buttons
        button_width = 200
        total_button_width = button_width * 2 + button_spacing
        start_x = center_x - total_button_width / 2
        start_y = 130

        # Update button text for toggling walls
        self.toggle_walls_top_bottom_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x, start_y, button_width, button_height),
            text="Add Walls Top/Bottom",
            manager=self.manager
        )

        self.clear_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + button_width + button_spacing, start_y, button_width,
                                      button_height),
            text="Clear All",
            manager=self.manager
        )

        self.toggle_walls_left_right_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x, start_y + button_height + button_spacing, button_width,
                                      button_height),
            text="Add Walls Left/Right",
            manager=self.manager
        )

        self.finish_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(start_x + button_width + button_spacing,
                                      start_y + button_height + button_spacing, button_width, button_height),
            text="Finish Setup",
            manager=self.manager
        )

        # Add right panel for growth region settings
        self.right_panel = pygame_gui.elements.UIPanel(
            relative_rect=pygame.Rect(self.screen_width - self.right_panel_width, 0, self.right_panel_width,
                                      self.screen_height),
            manager=self.manager
        )

        # Label for the right panel
        self.region_settings_label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(10, 10, 280, 30),
            text="Growth Region Settings",
            manager=self.manager,
            container=self.right_panel,
            object_id="#bold_label"
        )

        label_height = 20
        entry_height = 25
        spacing = 15
        start_y = 50
        label_width = 130
        entry_width = 170

        entries = [
            ('num_plants_entry', "Num Plants"),
            ('tau_min_entry', "Tau Min"),
            ('tau_max_entry', "Tau Max"),
            ('r_min_entry', "R Min"),
            ('r_max_entry', "R Max"),
            ('top_left_x_entry', "Top-Left X"),
            ('top_left_y_entry', "Top-Left Y"),
            ('bottom_right_x_entry', "Bottom-Right X"),
            ('bottom_right_y_entry', "Bottom-Right Y"),
            ('num_agents_entry', "Num Agents")
        ]

        for i, (attr_name, label_text) in enumerate(entries):
            y_pos = start_y + i * (label_height + entry_height + spacing)

            setattr(self, attr_name, pygame_gui.elements.UITextEntryLine(
                relative_rect=pygame.Rect(label_width + 10, y_pos, entry_width, entry_height),
                manager=self.manager,
                container=self.right_panel
            ))

            pygame_gui.elements.UILabel(
                relative_rect=pygame.Rect(10, y_pos, label_width, label_height),
                text=f"{label_text}:",
                manager=self.manager,
                container=self.right_panel
            )

        self.apply_settings_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect(10, start_y + 10 * (label_height + entry_height + spacing), 300, 30),
            text="Apply Settings",
            manager=self.manager,
            container=self.right_panel
        )

        self.set_right_panel_enabled(False)

    def run(self):
        running = True
        while running:
            time_delta = self.clock.tick(60) / 1000.0

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None

                if self.agents_move_checkbox.handle_event(event):
                    self.agents_can_move = self.agents_move_checkbox.is_checked

                if event.type == pygame.USEREVENT:
                    if event.user_type == pygame_gui.UI_TEXT_ENTRY_CHANGED:
                        if event.ui_element in self.hyperparameter_entries.values():
                            self.on_hyperparameter_changed(event)
                        elif event.ui_element in [self.width_entry, self.height_entry]:
                            self.on_size_changed(event)
                        elif event.ui_element == self.seed_entry:
                            self.seed = event.ui_element.get_text()
                        elif event.ui_element in [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry,
                                                  self.r_min_entry, self.r_max_entry, self.top_left_x_entry,
                                                  self.top_left_y_entry, self.bottom_right_x_entry,
                                                  self.bottom_right_y_entry, self.num_agents_entry]:
                            self.on_region_parameter_changed(event)
                        event.ui_element.redraw()

                    elif event.user_type == pygame_gui.UI_BUTTON_PRESSED:
                        if event.ui_element == self.apply_button:
                            self.apply_size()
                        elif event.ui_element == self.toggle_walls_top_bottom_button:
                            self.toggle_walls_top_bottom()
                        elif event.ui_element == self.toggle_walls_left_right_button:
                            self.toggle_walls_left_right()
                        elif event.ui_element == self.clear_button:
                            self.clear_all()
                        elif event.ui_element == self.finish_button:
                            return self.finish_setup()
                        elif event.ui_element == self.apply_settings_button:
                            self.apply_region_settings()
                        elif event.ui_element == self.save_config_button:
                            self.save_config()
                        elif event.ui_element == self.load_config_button:
                            self.load_config()
                        elif event.ui_element == self.random_seed_button:
                            self.seed = "RANDOM"
                            self.seed_entry.set_text("RANDOM")

                if event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1:  # Left mouse button
                        if self.is_within_grid(event.pos):
                            self.click_start_pos = event.pos
                        else:
                            self.click_start_pos = None
                    elif event.button == 3:  # Right mouse button
                        self.remove_region_at_position(event.pos)

                if event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:  # Left mouse button
                        if self.click_start_pos and self.is_within_grid(event.pos):
                            dx = event.pos[0] - self.click_start_pos[0]
                            dy = event.pos[1] - self.click_start_pos[1]
                            if (dx * dx + dy * dy) < self.click_threshold * self.click_threshold:
                                # This was a click, not a drag
                                self.select_region_at_position(event.pos)
                            else:
                                # This was a drag, finalize drawing
                                self.stop_drawing()
                        self.click_start_pos = None
                        self.drawing = False
                        self.current_region = None

                if event.type == pygame.MOUSEMOTION:
                    if event.buttons[0] and self.click_start_pos:  # Left mouse button is held down
                        if self.is_within_grid(event.pos):
                            dx = event.pos[0] - self.click_start_pos[0]
                            dy = event.pos[1] - self.click_start_pos[1]
                            if (dx * dx + dy * dy) >= self.click_threshold * self.click_threshold:
                                # Start drawing only if we've moved past the threshold
                                if not self.drawing:
                                    self.start_drawing(self.click_start_pos)
                                self.update_current_region(event.pos)
                        else:
                            # If we've moved outside the grid, stop drawing
                            self.drawing = False
                            self.current_region = None

                self.manager.process_events(event)

            self.manager.update(time_delta)
            self.draw()

        return None

    def draw(self):
        self.screen.fill(self.bg_color)

        grid_left, grid_top, grid_width, grid_height = self.get_grid_position()

        # Draw environment grid
        for x in range(self.environment_width + 1):
            start_pos = (grid_left + x * self.cell_size, grid_top)
            end_pos = (grid_left + x * self.cell_size, grid_top + grid_height)
            pygame.draw.line(self.screen, self.grid_color, start_pos, end_pos)
        for y in range(self.environment_height + 1):
            start_pos = (grid_left, grid_top + y * self.cell_size)
            end_pos = (grid_left + grid_width, grid_top + y * self.cell_size)
            pygame.draw.line(self.screen, self.grid_color, start_pos, end_pos)

        # Draw growth regions
        for i, region in enumerate(self.growth_regions):
            rect = pygame.Rect(
                grid_left + region.left * self.cell_size,
                grid_top + region.top * self.cell_size,
                region.width * self.cell_size,
                region.height * self.cell_size
            )
            if region == self.selected_region:
                pygame.draw.rect(self.screen, self.selected_region_border_color, rect)
                inner_rect = rect.inflate(-4, -4)
                pygame.draw.rect(self.screen, self.selected_region_fill_color, inner_rect)
            else:
                pygame.draw.rect(self.screen, self.growth_region_color, rect)

            self.draw_region_parameters(rect, self.growth_region_data[i])

        # Draw current region being drawn
        if self.current_region:
            pygame.draw.rect(self.screen, self.current_region_color, (
                grid_left + self.current_region.left * self.cell_size,
                grid_top + self.current_region.top * self.cell_size,
                self.current_region.width * self.cell_size,
                self.current_region.height * self.cell_size
            ))

        # Draw walls if toggled
        if self.walls_top_bottom or self.walls_left_right:
            wall_thickness = self.cell_size
            if self.walls_top_bottom:
                pygame.draw.rect(self.screen, self.wall_color, (grid_left, grid_top, grid_width, wall_thickness))
                pygame.draw.rect(self.screen, self.wall_color,
                                 (grid_left, grid_top + grid_height - wall_thickness, grid_width, wall_thickness))
            if self.walls_left_right:
                pygame.draw.rect(self.screen, self.wall_color, (grid_left, grid_top, wall_thickness, grid_height))
                pygame.draw.rect(self.screen, self.wall_color,
                                 (grid_left + grid_width - wall_thickness, grid_top, wall_thickness, grid_height))

        self.agents_move_checkbox.draw(self.screen)

        self.draw_instructions()
        self.manager.draw_ui(self.screen)
        pygame.display.flip()

    def save_config(self):
        """
        Save the interface configuration to setup_config.json in the current directory.
        """
        config = {
            'width': self.environment_width,
            'height': self.environment_height,
            'growth_regions': [
                {
                    'x_min': rect.left,
                    'y_min': rect.top,
                    'x_max': rect.right - 1,
                    'y_max': rect.bottom - 1,
                    **self.growth_region_data[i]
                }
                for i, rect in enumerate(self.growth_regions)
            ],
            'walls_top_bottom': self.walls_top_bottom,
            'walls_left_right': self.walls_left_right,
            'hyperparameters': self.get_hyperparameters(),
            'agents_can_move': self.agents_can_move,
            'seed': self.seed
        }

        try:
            with open('setup_config.json', 'w') as f:
                json.dump(config, f, indent=4)
            print("Configuration saved successfully.")
        except IOError as e:
            print(f"Error saving configuration: {e}")

    def load_config(self):
        if not os.path.exists('setup_config.json'):
            print("Configuration file not found.")
            return False

        try:
            with open('setup_config.json', 'r') as f:
                config = json.load(f)

            # Update environment size
            self.environment_width = config['width']
            self.environment_height = config['height']
            self.width_entry.set_text(str(config['width']))
            self.height_entry.set_text(str(config['height']))

            # Apply the new size
            self.apply_size()

            # Update growth regions
            self.growth_regions = []
            self.growth_region_data = []
            for region in config['growth_regions']:
                rect = pygame.Rect(region['x_min'], region['y_min'],
                                   region['x_max'] - region['x_min'] + 1,
                                   region['y_max'] - region['y_min'] + 1)
                self.growth_regions.append(rect)
                self.growth_region_data.append({
                    'num_plants': region['num_plants'],
                    'tau_min': region['tau_min'],
                    'tau_max': region['tau_max'],
                    'r_min': region['r_min'],
                    'r_max': region['r_max'],
                    'num_agents': region.get('num_agents', 0)
                })

            # Update walls
            self.walls_top_bottom = config['walls_top_bottom']
            self.walls_left_right = config['walls_left_right']
            self.toggle_walls_top_bottom_button.set_text(
                "Remove Walls Top/Bottom" if self.walls_top_bottom else "Add Walls Top/Bottom")
            self.toggle_walls_left_right_button.set_text(
                "Remove Walls Left/Right" if self.walls_left_right else "Add Walls Left/Right")

            # Update hyperparameters
            for param_name, value in config['hyperparameters'].items():
                if param_name in self.hyperparameter_entries:
                    if isinstance(value, float):
                        formatted_value = f"{value:.10f}".rstrip('0').rstrip('.')
                    else:
                        formatted_value = str(value)
                    self.hyperparameter_entries[param_name].set_text(formatted_value)

            # Update agents_can_move
            self.agents_can_move = config.get('agents_can_move', False)
            self.agents_move_checkbox.is_checked = self.agents_can_move

            # Update seed
            self.seed = config.get('seed', 'RANDOM')
            self.seed_entry.set_text(self.seed)

            # Validate all inputs
            self.validate_environment_size()
            self.validate_all_hyperparameters()
            self.update_finish_button_state()
            self.update_text_colors()

            # Disable the right panel
            self.selected_region = None
            self.clear_region_settings_panel()
            self.set_right_panel_enabled(False)

            # Force a redraw of the interface
            self.draw()

            print("Configuration loaded successfully.")
            return True
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            print(f"Error loading configuration: {e}")
            return False

    def get_hyperparameters(self):
        hyperparameters = {}
        for param_name, entry in self.hyperparameter_entries.items():
            value = entry.get_text()
            if value:  # Only include non-empty values
                try:
                    # Convert to float if possible, otherwise keep as string
                    hyperparameters[param_name] = float(value)
                except ValueError:
                    hyperparameters[param_name] = value
        return hyperparameters

    def draw_instructions(self):
        total_width = sum(self.instruction_font.size(instruction)[0] for instruction in self.instructions)
        total_width += 20 * (len(self.instructions) - 1)  # Add spacing between instructions

        start_x = (self.screen_width - total_width) // 2
        start_y = self.screen_height - self.instruction_height

        for instruction in self.instructions:
            text_surface = self.instruction_font.render(instruction, True, self.text_color)
            text_rect = text_surface.get_rect(centery=start_y)
            text_rect.left = start_x
            self.screen.blit(text_surface, text_rect)
            start_x += text_rect.width + 20

    def set_right_panel_enabled(self, enabled):
        for entry in [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry, self.r_min_entry,
                      self.r_max_entry, self.top_left_x_entry, self.top_left_y_entry,
                      self.bottom_right_x_entry, self.bottom_right_y_entry, self.num_agents_entry]:
            if enabled:
                entry.enable()
            else:
                entry.set_text("")
                entry.update(0)
                entry.disable()
        self.apply_settings_button.enable() if enabled else self.apply_settings_button.disable()

    def get_grid_position(self):
        grid_width = self.environment_width * self.cell_size
        grid_height = self.environment_height * self.cell_size
        grid_left = self.left_panel_width + (self.screen_width - self.left_panel_width - self.right_panel_width - grid_width) // 2
        grid_top = self.grid_top_margin + (self.screen_height - self.grid_top_margin - self.grid_bottom_margin - grid_height) // 2
        return grid_left, grid_top, grid_width, grid_height

    def is_within_grid(self, pos):
        grid_left, grid_top, grid_width, grid_height = self.get_grid_position()
        return (grid_left <= pos[0] <= grid_left + grid_width and
                grid_top <= pos[1] <= grid_top + grid_height)

    def start_drawing(self, pos):
        if self.is_within_grid(pos):
            grid_left, grid_top, _, _ = self.get_grid_position()
            self.drawing = True
            self.start_pos = ((pos[0] - grid_left) // self.cell_size, (pos[1] - grid_top) // self.cell_size)
            self.current_region = pygame.Rect(self.start_pos[0], self.start_pos[1], 1, 1)

    def stop_drawing(self):
        if self.drawing:
            self.drawing = False
            if self.current_region:
                self.growth_regions.append(self.current_region)
                self.growth_region_data.append({
                    'num_plants': 0,
                    'tau_min': 100,
                    'tau_max': 2500,
                    'r_min': 0.0,
                    'r_max': 8.0,
                    'num_agents': 0
                })
                # Automatically select the newly created region
                self.selected_region = self.current_region
                self.update_region_settings_panel()
                self.set_right_panel_enabled(True)
                self.current_region = None

    def update_current_region(self, pos):
        if self.is_within_grid(pos) and self.drawing:
            grid_left, grid_top, _, _ = self.get_grid_position()
            current_pos = ((pos[0] - grid_left) // self.cell_size, (pos[1] - grid_top) // self.cell_size)
            x1, y1 = self.start_pos
            x2, y2 = current_pos
            left = max(0, min(x1, x2))
            top = max(0, min(y1, y2))
            width = min(self.environment_width - left, abs(x2 - x1) + 1)
            height = min(self.environment_height - top, abs(y2 - y1) + 1)
            self.current_region = pygame.Rect(left, top, width, height)

    def select_region_at_position(self, pos):
        grid_left, grid_top, _, _ = self.get_grid_position()
        mouse_x = (pos[0] - grid_left) // self.cell_size
        mouse_y = (pos[1] - grid_top) // self.cell_size

        for region in self.growth_regions:
            if region.collidepoint(mouse_x, mouse_y):
                self.selected_region = region
                self.clear_region_settings_panel()  # This now clears invalid_fields
                self.set_right_panel_enabled(True)
                self.update_region_settings_panel()
                return

        self.selected_region = None
        self.clear_region_settings_panel()
        self.set_right_panel_enabled(False)

    def update_region_settings_panel(self):
        if self.selected_region:
            index = self.growth_regions.index(self.selected_region)
            region_data = self.growth_region_data[index]
            self.num_plants_entry.set_text(str(region_data['num_plants']))
            self.tau_min_entry.set_text(str(region_data['tau_min']))
            self.tau_max_entry.set_text(str(region_data['tau_max']))
            self.r_min_entry.set_text(f"{region_data['r_min']:.1f}")
            self.r_max_entry.set_text(f"{region_data['r_max']:.1f}")
            self.top_left_x_entry.set_text(str(self.selected_region.left))
            self.top_left_y_entry.set_text(str(self.selected_region.top))
            self.bottom_right_x_entry.set_text(str(self.selected_region.right - 1))
            self.bottom_right_y_entry.set_text(str(self.selected_region.bottom - 1))
            self.num_agents_entry.set_text(str(region_data.get('num_agents', 0)))

            # Validate the newly set values
            self.validate_region_parameters()
            self.update_text_colors()

    def clear_region_settings_panel(self):
        for entry in [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry, self.r_min_entry,
                      self.r_max_entry, self.top_left_x_entry, self.top_left_y_entry,
                      self.bottom_right_x_entry, self.bottom_right_y_entry, self.num_agents_entry]:
            entry.set_text("")
            self.invalid_fields.discard(entry)  # Remove the entry from invalid_fields

        # Clear all invalid fields related to the right panel
        self.invalid_fields = {field for field in self.invalid_fields
                               if field not in [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry,
                                                self.r_min_entry, self.r_max_entry, self.top_left_x_entry,
                                                self.top_left_y_entry, self.bottom_right_x_entry,
                                                self.bottom_right_y_entry, self.num_agents_entry]}

        # Update the text colors for all entries
        self.update_text_colors()

    def apply_region_settings(self):
        if self.selected_region:
            index = self.growth_regions.index(self.selected_region)
            try:
                new_left = int(self.top_left_x_entry.get_text())
                new_top = int(self.top_left_y_entry.get_text())
                new_right = int(self.bottom_right_x_entry.get_text()) + 1
                new_bottom = int(self.bottom_right_y_entry.get_text()) + 1

                # Ensure the new coordinates are within the environment bounds
                new_left = max(0, min(new_left, self.environment_width - 1))
                new_top = max(0, min(new_top, self.environment_height - 1))
                new_right = max(new_left + 1, min(new_right, self.environment_width))
                new_bottom = max(new_top + 1, min(new_bottom, self.environment_height))

                self.selected_region.update(new_left, new_top, new_right - new_left, new_bottom - new_top)

                self.growth_region_data[index] = {
                    'num_plants': int(self.num_plants_entry.get_text()),
                    'tau_min': int(self.tau_min_entry.get_text()),
                    'tau_max': int(self.tau_max_entry.get_text()),
                    'r_min': float(self.r_min_entry.get_text()),
                    'r_max': float(self.r_max_entry.get_text()),
                    'num_agents': int(self.num_agents_entry.get_text())
                }
            except ValueError:
                print("Invalid input. Please enter valid numbers for all fields.")

    def remove_region_at_position(self, pos):
        grid_left, grid_top, _, _ = self.get_grid_position()
        mouse_x = (pos[0] - grid_left) // self.cell_size
        mouse_y = (pos[1] - grid_top) // self.cell_size

        for i, region in enumerate(self.growth_regions):
            if region.collidepoint(mouse_x, mouse_y):
                del self.growth_regions[i]
                del self.growth_region_data[i]
                if self.selected_region == region:
                    self.selected_region = None
                    self.clear_region_settings_panel()
                    self.set_right_panel_enabled(False)
                break

    def apply_size(self):
        try:
            new_width = int(self.width_entry.get_text())
            new_height = int(self.height_entry.get_text())

            self.environment_width = new_width
            self.environment_height = new_height
            self.cell_size = min(
                (self.screen_width - self.left_panel_width - self.right_panel_width) // self.environment_width,
                (self.screen_height - self.grid_top_margin - self.grid_bottom_margin) // self.environment_height
            )

            self.growth_regions = []
            self.walls_top_bottom = False
            self.walls_left_right = False
        except ValueError:
            print("Invalid input. Please enter valid integers for width and height.")

    def toggle_walls_top_bottom(self):
        self.walls_top_bottom = not self.walls_top_bottom
        button_text = "Remove Walls Top/Bottom" if self.walls_top_bottom else "Add Walls Top/Bottom"
        self.toggle_walls_top_bottom_button.set_text(button_text)

    def toggle_walls_left_right(self):
        self.walls_left_right = not self.walls_left_right
        button_text = "Remove Walls Left/Right" if self.walls_left_right else "Add Walls Left/Right"
        self.toggle_walls_left_right_button.set_text(button_text)

    def clear_all(self):
        self.growth_regions = []
        self.walls_top_bottom = False
        self.walls_left_right = False
        self.toggle_walls_top_bottom_button.set_text("Add Walls Top/Bottom")
        self.toggle_walls_left_right_button.set_text("Add Walls Left/Right")
        self.selected_region = None
        self.clear_region_settings_panel()
        self.set_right_panel_enabled(False)

    def draw_region_parameters(self, rect, region_data):
        lines = [
            f"Plants: {region_data['num_plants']}",
            f"Tau: {region_data['tau_min']}-{region_data['tau_max']}",
            f"R: {region_data['r_min']:.1f}-{region_data['r_max']:.1f}",
            f"Agents: {region_data.get('num_agents', 0)}"
        ]

        line_height = self.font.get_linesize()
        total_height = line_height * len(lines)

        start_y = rect.centery - total_height // 2

        for i, line in enumerate(lines):
            text_surface = self.font.render(line, True, (255, 255, 255))  # White text
            text_rect = text_surface.get_rect(center=(rect.centerx, start_y + i * line_height))
            self.screen.blit(text_surface, text_rect)

    def finish_setup(self):
        all_params = {
            'width': self.environment_width,
            'height': self.environment_height,
            'growth_regions': [
                {
                    'x_min': rect.left,
                    'y_min': rect.top,
                    'x_max': rect.right - 1,
                    'y_max': rect.bottom - 1,
                    **self.growth_region_data[i]
                }
                for i, rect in enumerate(self.growth_regions)
            ],
            'walls_top_bottom': self.walls_top_bottom,
            'walls_left_right': self.walls_left_right,
            'hyperparameters': self.get_hyperparameters(),
            'agents_can_move': self.agents_can_move,
            'seed': self.seed
        }
        self.save_config()
        return all_params

    def convert_rect_to_grid_coords(self, rect):
        return {
            'x_min': rect.left,
            'y_min': rect.top,
            'x_max': rect.right - 1,
            'y_max': rect.bottom - 1
        }

    def on_hyperparameter_changed(self, event):
        self.validate_all_hyperparameters()
        self.update_finish_button_state()
        self.update_text_colors()

    def on_size_changed(self, event):
        self.validate_environment_size()
        self.update_finish_button_state()
        self.update_text_colors()

    def on_region_parameter_changed(self, event):
        self.validate_region_parameters()
        self.update_apply_settings_button_state()
        self.update_text_colors()

    def update_finish_button_state(self):
        if len(self.invalid_fields) == 0:
            self.finish_button.enable()
        else:
            self.finish_button.disable()

    def update_apply_settings_button_state(self):
        if len(self.invalid_fields.intersection(set([self.num_plants_entry, self.tau_min_entry, self.tau_max_entry,
                                                     self.r_min_entry, self.r_max_entry, self.top_left_x_entry,
                                                     self.top_left_y_entry, self.bottom_right_x_entry,
                                                     self.bottom_right_y_entry, self.num_agents_entry]))) == 0:
            self.apply_settings_button.enable()
        else:
            self.apply_settings_button.disable()

    def update_text_colors(self):
        for entry in self.hyperparameter_entries.values():
            self.set_text_color(entry)
        self.set_text_color(self.width_entry)
        self.set_text_color(self.height_entry)
        for entry in [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry, self.r_min_entry,
                      self.r_max_entry, self.top_left_x_entry, self.top_left_y_entry,
                      self.bottom_right_x_entry, self.bottom_right_y_entry, self.num_agents_entry]:
            if not entry.is_enabled:  # Only update colors for enabled entries
                continue
            self.set_text_color(entry)

    def set_text_color(self, entry):
        if not entry.is_enabled:  # Don't change colors for disabled entries
            return
        if entry in self.invalid_fields:
            entry.text_colour = pygame.Color('#FF0000')  # Red color for invalid input
            entry.selected_text_colour = pygame.Color('#FF0000')
        else:
            entry.text_colour = pygame.Color('#FFFFFF')  # White color for valid input
            entry.selected_text_colour = pygame.Color('#FFFFFF')
        entry.rebuild()

    def validate_environment_size(self):
        self.invalid_fields.discard(self.width_entry)
        self.invalid_fields.discard(self.height_entry)
        try:
            width = int(self.width_entry.get_text())
            height = int(self.height_entry.get_text())
            if not (0 < width <= 600 and 0 < height <= 600):
                self.invalid_fields.add(self.width_entry)
                self.invalid_fields.add(self.height_entry)
        except ValueError:
            self.invalid_fields.add(self.width_entry)
            self.invalid_fields.add(self.height_entry)

    def validate_region_parameters(self):
        region_entries = [self.num_plants_entry, self.tau_min_entry, self.tau_max_entry, self.r_min_entry,
                          self.r_max_entry, self.top_left_x_entry, self.top_left_y_entry,
                          self.bottom_right_x_entry, self.bottom_right_y_entry, self.num_agents_entry]
        for entry in region_entries:
            self.invalid_fields.discard(entry)

        try:
            width = int(self.width_entry.get_text())
            height = int(self.height_entry.get_text())
            x_min = int(self.top_left_x_entry.get_text())
            y_min = int(self.top_left_y_entry.get_text())
            x_max = int(self.bottom_right_x_entry.get_text())
            y_max = int(self.bottom_right_y_entry.get_text())
            num_plants = int(self.num_plants_entry.get_text())
            tau_min = int(self.tau_min_entry.get_text())
            tau_max = int(self.tau_max_entry.get_text())
            r_min = float(self.r_min_entry.get_text())
            r_max = float(self.r_max_entry.get_text())
            num_agents = int(self.num_agents_entry.get_text())

            # Validate x_min, y_min
            if not (0 <= x_min < width):
                self.invalid_fields.add(self.top_left_x_entry)
            if not (0 <= y_min < height):
                self.invalid_fields.add(self.top_left_y_entry)

            # Validate x_max, y_max
            if not (x_min < x_max < width):
                self.invalid_fields.add(self.bottom_right_x_entry)
            if not (y_min < y_max < height):
                self.invalid_fields.add(self.bottom_right_y_entry)

            # Validate num_plants
            region_area = (x_max - x_min + 1) * (y_max - y_min + 1)
            if not (0 < num_plants <= region_area):
                self.invalid_fields.add(self.num_plants_entry)

            # Validate tau_min, tau_max
            if not (0 <= tau_min <= tau_max):
                self.invalid_fields.add(self.tau_min_entry)
                self.invalid_fields.add(self.tau_max_entry)

            # Validate r_min, r_max
            if not (0 <= r_min <= r_max):
                self.invalid_fields.add(self.r_min_entry)
                self.invalid_fields.add(self.r_max_entry)

            # Validate num_agents
            if num_agents < 0:
                self.invalid_fields.add(self.num_agents_entry)

        except ValueError:
            for entry in region_entries:
                self.invalid_fields.add(entry)

    def validate_all_hyperparameters(self):
        for param_name, entry in self.hyperparameter_entries.items():
            if not self.validate_hyperparameter(param_name, entry.get_text()):
                self.invalid_fields.add(entry)
            else:
                self.invalid_fields.discard(entry)

    def validate_hyperparameter(self, param_name, value):
        try:
            float_value = float(value)
            if param_name in ['Basal Metabolic Cost', 'Neuron Cost', 'Synapse Cost', 'Replication Cost']:
                return float_value >= 0
            elif param_name in ['Consumption Rate', 'Transfer Rate']:
                return float_value > 0
            elif param_name in ['Replication Survivability Scalar', 'Replication Cooldown',
                                'Number of Communication Signals']:
                return float_value >= 0 and float_value.is_integer()
            elif param_name in ['Progenitor Update Threshold']:
                return float_value > 0 and float_value.is_integer()
            elif param_name in ['Crossover Probability', 'Mutation Probability', 'Compatibility Threshold']:
                return 0 < float_value < 1
            else:
                return True  # Unknown parameter, assume it's valid
        except ValueError:
            return False


class CustomCheckbox:
    def __init__(self, x, y, height, label, manager, text_checkbox_distance=15, text_color=(200, 200, 200),
                 button_color='#3a3a3a', active_color='#5a5a5a', tick_color='#ffffff'):
        self.label = label
        self.manager = manager
        self.text_color = pygame.Color(text_color)
        self.button_color = pygame.Color(button_color)
        self.active_color = pygame.Color(active_color)
        self.tick_color = pygame.Color(tick_color)
        self.is_checked = False
        self.text_checkbox_distance = text_checkbox_distance

        # Create a label using pygame_gui
        self.label_element = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect(0, y, -1, height),  # We'll set the x-coordinate later
            text=self.label,
            manager=self.manager
        )

        # Get the size of the label
        label_width = self.label_element.rect.width
        label_height = self.label_element.rect.height

        # Calculate the checkbox button size
        self.button_size = min(height - 4, 20)  # Limit button size to height - 4 or 20, whichever is smaller

        # Calculate the total width of the label, gap, and checkbox
        self.total_width = label_width + self.text_checkbox_distance + self.button_size

        # Adjust the x position so that the label + checkbox is centered at the given x position
        label_x = x - self.total_width // 2

        # Set the label position to be centered vertically with respect to the checkbox
        self.label_element.set_position((label_x, y + (height - label_height) // 2))

        # Create the checkbox rect next to the label with the specified distance
        self.checkbox_rect = pygame.Rect(label_x + label_width + self.text_checkbox_distance, y, self.button_size, self.button_size)

        # Create the actual rect for the entire checkbox (used for event handling)
        self.rect = pygame.Rect(label_x, y, self.total_width, height)

        # Position the checkbox (button) on the right side of the label
        self.button_rect = pygame.Rect(label_x + label_width + self.text_checkbox_distance,
                                       y + (height - self.button_size) // 2,
                                       self.button_size, self.button_size)

    def draw(self, surface):
        # Draw the button (checkbox)
        pygame.draw.rect(surface, self.active_color if self.is_checked else self.button_color, self.button_rect)
        pygame.draw.rect(surface, pygame.Color('#1c1c1c'), self.button_rect, 1)  # border

        # Draw the tick if checked
        if self.is_checked:
            pygame.draw.line(surface, self.tick_color,
                             (self.button_rect.left + 4, self.button_rect.centery),
                             (self.button_rect.centerx, self.button_rect.bottom - 4), 3)
            pygame.draw.line(surface, self.tick_color,
                             (self.button_rect.centerx, self.button_rect.bottom - 4),
                             (self.button_rect.right - 4, self.button_rect.top + 4), 3)

    def toggle(self):
        self.is_checked = not self.is_checked

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.toggle()
                return True
        return False