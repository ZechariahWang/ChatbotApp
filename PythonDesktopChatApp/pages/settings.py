import customtkinter as ctk

class SettingsPage(ctk.CTkFrame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_rowconfigure((0, 2), weight=1)
        self.grid_rowconfigure(1, weight=0)
        self.grid_columnconfigure((0, 2), weight=1)
        self.grid_columnconfigure(1, weight=1)

        label = ctk.CTkLabel(self, text="Settings Page", font=("Arial", 20))
        label.grid(row=1, column=1, pady=(20, 10))

        back_button = ctk.CTkButton(self, text="← Home", command=lambda: controller.show_page("HomePage"))
        back_button.grid(row=2, column=1, pady=(10, 20))
