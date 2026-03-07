import sys
import os
sys.path.insert(0, os.getcwd())
from src.extraction.common.content_recognizer import ContentRecognizer
cr = ContentRecognizer()
print(cr.molnextr)
