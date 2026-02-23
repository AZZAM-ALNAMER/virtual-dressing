"""
Template tags for i18n: build language-prefixed path for language switcher redirect.
"""
from django import template
from django.conf import settings
from django.utils.translation import get_language

register = template.Library()


@register.simple_tag(takes_context=True)
def path_for_lang(context, lang_code):
    """
    Return the current path with the given language prefix for use as 'next' in set_language.
    With prefix_default_language=False: default (en) has no prefix, ar has /ar/.
    """
    request = context.get('request')
    if not request:
        return '/'
    path = request.get_full_path()
    # Strip existing language prefix (e.g. /ar/ or /en/ if ever used)
    path_without_prefix = path
    for code, _ in settings.LANGUAGES:
        prefix = f'/{code}/'
        if path.startswith(prefix):
            path_without_prefix = path[len(prefix):] or '/'
            break
        if path == f'/{code}' or path == f'/{code}/':
            path_without_prefix = '/'
            break
    if not path_without_prefix.startswith('/'):
        path_without_prefix = '/' + path_without_prefix
    if path_without_prefix == '//':
        path_without_prefix = '/'
    default_lang = settings.LANGUAGE_CODE
    if lang_code == default_lang:
        return path_without_prefix
    return f'/{lang_code}{path_without_prefix}' if path_without_prefix != '/' else f'/{lang_code}/'
