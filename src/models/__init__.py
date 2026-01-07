from .paper import Paper

__all__ = [
    "Paper",
]

"""
The __init__.py file serves two main purposes in Python:
1. Marks a Directory as a Package: It tells Python that the directory contains Python code and should be treated as a package, allowing you to import modules from it (e.g., import src.models).
2. Exposes API: It can import key classes or functions from internal modules and expose them at the package level, simplifying imports for users.
"""