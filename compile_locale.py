#!/usr/bin/env python
"""
Compile Django .po files to .mo when GNU gettext (msgfmt) is not installed (e.g. Windows).
Usage: python compile_locale.py
Requires: pip install babel
"""
import os
import sys

try:
    from babel.messages.pofile import read_po
    from babel.messages.mofile import write_mo
except ImportError:
    print("Install babel: pip install babel")
    sys.exit(1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCALE_DIR = os.path.join(BASE_DIR, "locale")

for lang in ("en", "ar"):
    po_path = os.path.join(LOCALE_DIR, lang, "LC_MESSAGES", "django.po")
    mo_path = os.path.join(LOCALE_DIR, lang, "LC_MESSAGES", "django.mo")
    if not os.path.isfile(po_path):
        continue
    with open(po_path, "rb") as f:
        catalog = read_po(f, locale=lang)
    with open(mo_path, "wb") as f:
        write_mo(f, catalog)
    print(f"Compiled: {mo_path}")

print("Done.")
