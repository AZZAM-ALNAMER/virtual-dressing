"""
Unified Color Palettes
Single source of truth for every colour used in the store, avatar, and AI prompts.

Structure:
    SKIN_TONE_PALETTES[skin_tone]['shirts']  → list of 4 dicts  {name, hex}
    SKIN_TONE_PALETTES[skin_tone]['pants']   → list of 4 dicts  {name, hex}

Rules:
    • Gemini MUST pick from the 4 shirt colours + 4 pants colours for the
      detected skin tone — nothing else.
    • The database seeder creates Color + ProductVariant rows for every
      colour listed here.
    • The 3D avatar renders exactly these swatches.
"""

SKIN_TONE_PALETTES = {
    # ───────────────────────────────────────────────
    'very_light': {
        'shirts': [
            {'name': 'Lavender',   'hex': '#e8d5f5'},
            {'name': 'Baby Blue',  'hex': '#b5d8f7'},
            {'name': 'Blush Pink', 'hex': '#f7c5d0'},
            {'name': 'Charcoal',   'hex': '#37474f'},
        ],
        'pants': [
            {'name': 'Ash Grey', 'hex': '#cfd8dc'},
            {'name': 'Slate',    'hex': '#607d8b'},
            {'name': 'Indigo',   'hex': '#1a237e'},
            {'name': 'Black',    'hex': '#111111'},
        ],
    },
    # ───────────────────────────────────────────────
    'light': {
        'shirts': [
            {'name': 'Sky Blue',   'hex': '#bbdefb'},
            {'name': 'Mint Green', 'hex': '#a5d6a7'},
            {'name': 'Rose',       'hex': '#ef9a9a'},
            {'name': 'White',      'hex': '#ffffff'},
        ],
        'pants': [
            {'name': 'Khaki',     'hex': '#c3b091'},
            {'name': 'Walnut',    'hex': '#6d4c41'},
            {'name': 'Deep Navy', 'hex': '#0d47a1'},
            {'name': 'Black',     'hex': '#111111'},
        ],
    },
    # ───────────────────────────────────────────────
    'intermediate': {
        'shirts': [
            {'name': 'Off-White',     'hex': '#f5f5f5'},
            {'name': 'Ocean Blue',    'hex': '#0277bd'},
            {'name': 'Olive Green',   'hex': '#558b2f'},
            {'name': 'Burnt Orange',  'hex': '#e64a19'},
        ],
        'pants': [
            {'name': 'Sand',       'hex': '#d4c5a9'},
            {'name': 'Dark Slate', 'hex': '#263238'},
            {'name': 'Cobalt',     'hex': '#1565c0'},
            {'name': 'Burgundy',   'hex': '#880e4f'},
        ],
    },
    # ───────────────────────────────────────────────
    'tan': {
        'shirts': [
            {'name': 'Cream',      'hex': '#fffde7'},
            {'name': 'Navy',       'hex': '#01579b'},
            {'name': 'Deep Green', 'hex': '#004d40'},
            {'name': 'Amber',      'hex': '#f57f17'},
        ],
        'pants': [
            {'name': 'Mocha',     'hex': '#8d6e63'},
            {'name': 'Mahogany',  'hex': '#3e2723'},
            {'name': 'Dark Teal', 'hex': '#006064'},
            {'name': 'Graphite',  'hex': '#37474f'},
        ],
    },
    # ───────────────────────────────────────────────
    'dark': {
        'shirts': [
            {'name': 'Lemon',         'hex': '#fff176'},
            {'name': 'Bright Orange', 'hex': '#ff6d00'},
            {'name': 'Electric Blue', 'hex': '#00e5ff'},
            {'name': 'Hot Pink',      'hex': '#ff4081'},
        ],
        'pants': [
            {'name': 'Pearl',        'hex': '#e0e0e0'},
            {'name': 'Silver',       'hex': '#bdbdbd'},
            {'name': 'Plum',         'hex': '#4a148c'},
            {'name': 'Almost Black', 'hex': '#212121'},
        ],
    },
}


def get_all_unique_colors():
    """Return a deduplicated list of all colour dicts across every skin tone."""
    seen = set()
    unique = []
    for palette in SKIN_TONE_PALETTES.values():
        for c in palette['shirts'] + palette['pants']:
            if c['name'] not in seen:
                seen.add(c['name'])
                unique.append(c)
    return unique


def get_shirt_colors(skin_tone: str):
    """Return the 4 shirt colour dicts for a given skin tone."""
    return SKIN_TONE_PALETTES.get(skin_tone, SKIN_TONE_PALETTES['intermediate'])['shirts']


def get_pants_colors(skin_tone: str):
    """Return the 4 pants colour dicts for a given skin tone."""
    return SKIN_TONE_PALETTES.get(skin_tone, SKIN_TONE_PALETTES['intermediate'])['pants']


def get_shirt_color_names(skin_tone: str):
    """Return a plain list of shirt colour names for a given skin tone."""
    return [c['name'] for c in get_shirt_colors(skin_tone)]


def get_pants_color_names(skin_tone: str):
    """Return a plain list of pants colour names for a given skin tone."""
    return [c['name'] for c in get_pants_colors(skin_tone)]
