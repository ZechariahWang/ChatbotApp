import customtkinter as ctk
from PIL import Image

class HomePage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.grid_rowconfigure((0, 5), weight=1)
        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_columnconfigure(1, weight=0)

        row = 1  

        try:
            image = ctk.CTkImage(
                light_image=Image.open("assets/logo.png"),
                dark_image=Image.open("assets/logo.png"),
                size=(100, 100)
            )
            image_label = ctk.CTkLabel(self, image=image, text="")
            image_label.grid(row=row, column=1, pady=10)
            row += 1
        except Exception as e:
            print(f"Error loading image: {e}")


        label = ctk.CTkLabel(self, text="Chatbot", font=("Roboto", 24, "bold"))
        label.grid(row=row, column=1, pady=(0, 20))
        row += 1

        chat_button = ctk.CTkButton(self, text="Go to Chat", command=lambda: controller.show_page("ChatPage"))
        chat_button.grid(row=row, column=1, pady=10)
        row += 1

        settings_button = ctk.CTkButton(self, text="Settings", command=lambda: controller.show_page("SettingsPage"))
        settings_button.grid(row=row, column=1, pady=10)
