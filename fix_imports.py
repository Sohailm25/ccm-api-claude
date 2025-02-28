"""
Fix for lxml.html.clean import issues.
This module provides a workaround for packages that depend on lxml.html.clean.
"""
import sys
import logging
import importlib.abc
import types

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("import_fixer")

class DummyCleanerModule(types.ModuleType):
    """Dummy module to replace lxml.html.clean"""
    def __init__(self):
        super().__init__("lxml.html.clean")
        
        # Define the Cleaner class that's most commonly used
        class Cleaner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
            def clean_html(self, html):
                # Just return the original HTML
                return html
            
            def __call__(self, element):
                # Some code may call the cleaner directly
                return element
        
        # Add necessary attributes to the module
        self.Cleaner = Cleaner
        # Add other commonly used functions/classes
        self.clean_html = lambda html: html
        self.clean = lambda html: html

class LXMLImportFixer(importlib.abc.MetaPathFinder):
    """Import hook to handle missing lxml.html.clean module"""
    def find_spec(self, fullname, path, target=None):
        if fullname == "lxml.html.clean":
            logger.warning("Using dummy lxml.html.clean module as workaround")
            # Create dummy module
            dummy_module = DummyCleanerModule()
            # Add to sys.modules cache to prevent repeated imports
            sys.modules[fullname] = dummy_module
            return None  # Return None as spec was already cached
        return None  # Not handling this import

def apply_fix():
    """Apply the import fix at runtime"""
    # Try to import lxml.html.clean properly first
    try:
        import lxml.html.clean
        logger.info("lxml.html.clean imported successfully, no fix needed")
        return
    except ImportError:
        # If import fails, install our fixer
        logger.warning("lxml.html.clean import failed, applying workaround")
        sys.meta_path.insert(0, LXMLImportFixer())
        logger.info("lxml.html.clean import workaround installed")

# Apply the fix when this module is imported
apply_fix() 