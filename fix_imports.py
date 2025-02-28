"""
Emergency workaround for lxml.html.clean import issue.
This file patches the import system to handle the missing module.
"""
import sys
import importlib.abc
import types
import logging

class DummyHTMLCleanModule(types.ModuleType):
    """Dummy module to replace lxml.html.clean"""
    def __init__(self):
        super().__init__("lxml.html.clean")
        
        # Define minimum required classes/functions
        class Cleaner:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                
            def clean_html(self, html):
                return html
                
        self.Cleaner = Cleaner

class LXMLImportFixer(importlib.abc.MetaPathFinder):
    """Import hook to handle missing lxml.html.clean module"""
    def find_spec(self, fullname, path, target=None):
        if fullname == "lxml.html.clean":
            logging.warning("Using dummy lxml.html.clean module as workaround")
            dummy_module = DummyHTMLCleanModule()
            sys.modules[fullname] = dummy_module
            return None
        return None

# Install the import hook
sys.meta_path.insert(0, LXMLImportFixer())
logging.info("Installed lxml.html.clean import workaround") 