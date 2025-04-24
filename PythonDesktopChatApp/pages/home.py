import customtkinter as ctk
from PIL import Image

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        # Configure grid
        self.grid_rowconfigure((0, 1, 2, 3), weight=1)
        self.grid_columnconfigure(0, weight=1)
        
        # Create main content frame
        self.content_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.content_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=20)
        self.content_frame.grid_rowconfigure((0, 1, 2), weight=1)
        self.content_frame.grid_columnconfigure(0, weight=1)

        # Add welcome text
        welcome_label = ctk.CTkLabel(
            self.content_frame,
            text="Mental Support Chatbot",
            font=("Roboto", 32, "bold"),
            text_color=("#2B2B2B", "#FFFFFF")
        )
        welcome_label.grid(row=0, column=0, pady=(0, 20))

        # Add description
        description_label = ctk.CTkLabel(
            self.content_frame,
            text="Personal AI assistant to help you with your mental health.\nStart a conversation or customize your experience.",
            font=("Roboto", 16),
            text_color=("#2B2B2B", "#FFFFFF"),
            justify="center"
        )
        description_label.grid(row=1, column=0, pady=(0, 40))

        # Create buttons frame
        buttons_frame = ctk.CTkFrame(self.content_frame, fg_color="transparent")
        buttons_frame.grid(row=2, column=0, sticky="nsew")
        buttons_frame.grid_columnconfigure((0, 1), weight=1)

        # Add chat button
        chat_button = ctk.CTkButton(
            buttons_frame,
            text="Start Chatting",
            command=lambda: controller.show_page("ChatPage"),
            height=50,
            font=("Roboto", 16, "bold"),
            corner_radius=10
        )
        chat_button.grid(row=0, column=0, padx=10, pady=10, sticky="ew")

        # Add settings button
        settings_button = ctk.CTkButton(
            buttons_frame,
            text="Settings",
            command=lambda: controller.show_page("SettingsPage"),
            height=50,
            font=("Roboto", 16, "bold"),
            corner_radius=10,
            fg_color="transparent",
            border_width=2
        )
        settings_button.grid(row=0, column=1, padx=10, pady=10, sticky="ew")
