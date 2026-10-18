"""PayRadar: Local-first late-payment radar for freelancers."""

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

__version__ = "0.1.0"
