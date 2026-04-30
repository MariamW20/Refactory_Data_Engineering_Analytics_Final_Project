from dashboard import demo
import os


if __name__ == "__main__":
    demo.launch(
    theme=gr.themes.Soft(),
    css=custom_css,
    share=False,
    server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    server_port=7861
)