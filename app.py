from dashboard import demo
import os


demo.launch(
    theme=gr.themes.Soft(),
    css=custom_css,
    share=False,
    server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
    server_port=int(os.getenv("PORT", "7860"))
)