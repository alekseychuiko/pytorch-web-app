"""App entry point."""
from flask_pytorch_web_app import create_app
import torch.multiprocessing as mp

app = create_app()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    app.run()
