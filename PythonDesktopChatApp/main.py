import customtkinter as ctk

# Import your page classes
from pages.home import HomePage
from pages.chat import ChatPage
from pages.settings import SettingsPage

ctk.set_appearance_mode("System") 
ctk.set_default_color_theme("green") 

class ChatbotApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Modular Chatbot")
        self.geometry("800x600")
        self.resizable(True, True)

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill="both", expand=True)

        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        self.pages = {}

        for PageClass in (HomePage, ChatPage, SettingsPage):
            page_name = PageClass.__name__
            frame = PageClass(parent=self.container, controller=self)
            self.pages[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_page("HomePage")

    def show_page(self, page_name):
        self.pages[page_name].tkraise()

if __name__ == "__main__":
    app = ChatbotApp()
    app.mainloop()
