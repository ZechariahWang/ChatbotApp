import customtkinter as ctk
from PIL import Image

# Import your page classes
from pages.home import HomePage
from pages.chat import ChatPage
from pages.settings import SettingsPage

ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("blue") 

class ChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Modular Chatbot")
        self.geometry("1000x700")
        self.minsize(800, 600)
        self.resizable(True, True)

        # Create main container
        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)

        # Create sidebar
        self.sidebar = ctk.CTkFrame(self.container, width=200, corner_radius=0)
        self.sidebar.pack(side="left", fill="y")
        self.sidebar.pack_propagate(False)

        # Add logo to sidebar
        # try:
        #     logo = ctk.CTkImage(
        #         light_image=Image.open("assets/logo.png"),
        #         dark_image=Image.open("assets/logo.png"),
        #         size=(150, 150)
        #     )
        #     logo_label = ctk.CTkLabel(self.sidebar, image=logo, text="")
        #     logo_label.pack(pady=20)
        # except Exception as e:
        #     print(f"Error loading logo: {e}")

        # Add navigation buttons
        self.home_button = ctk.CTkButton(
            self.sidebar, 
            text="Home", 
            command=lambda: self.show_page("HomePage"),
            height=40,
            corner_radius=10
        )
        self.home_button.pack(pady=10, padx=20, fill="x")

        self.chat_button = ctk.CTkButton(
            self.sidebar, 
            text="Chat", 
            command=lambda: self.show_page("ChatPage"),
            height=40,
            corner_radius=10
        )
        self.chat_button.pack(pady=10, padx=20, fill="x")

        self.settings_button = ctk.CTkButton(
            self.sidebar, 
            text="Settings", 
            command=lambda: self.show_page("SettingsPage"),
            height=40,
            corner_radius=10
        )
        self.settings_button.pack(pady=10, padx=20, fill="x")

        # Create content frame
        self.content_frame = ctk.CTkFrame(self.container)
        self.content_frame.pack(side="right", fill="both", expand=True)

        self.pages = {}
        for PageClass in (HomePage, ChatPage, SettingsPage):
            page_name = PageClass.__name__
            frame = PageClass(parent=self.content_frame, controller=self)
            self.pages[page_name] = frame
            frame.pack(fill="both", expand=True)

        self.show_page("HomePage")

    def show_page(self, page_name):
        for page in self.pages.values():
            page.pack_forget()
        self.pages[page_name].pack(fill="both", expand=True)

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()
