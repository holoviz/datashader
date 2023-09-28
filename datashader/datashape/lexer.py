"""
Lexer for the datashape grammar.
"""

from __future__ import absolute_import, division, print_function

import re
import ast
import collections

from . import error

# This is updated to include all the token names from _tokens,
# where e.g. _tokens[NAME_LOWER-1] is the entry for NAME_LOWER
__all__ = ['lex', 'Token']

def _str_val(s):
    # Use the Python parser via the ast module to parse the string,
    # since the string_escape and unicode_escape codecs do the wrong thing
    return ast.parse('u' + s).body[0].value.s

# A list of the token names, corresponding regex, and value extraction function
_tokens = [
    ('BOOLEAN',    r'True|False', ast.literal_eval),
    ('NAME_LOWER', r'[a-z][a-zA-Z0-9_]*', lambda x : x),
    ('NAME_UPPER', r'[A-Z][a-zA-Z0-9_]*', lambda x : x),
    ('NAME_OTHER', r'_[a-zA-Z0-9_]*', lambda x : x),
    ('ASTERISK',   r'\*'),
    ('COMMA',      r','),
    ('EQUAL',      r'='),
    ('COLON',      r':'),
    ('LBRACKET',   r'\['),
    ('RBRACKET',   r'\]'),
    ('LBRACE',     r'\{'),
    ('RBRACE',     r'\}'),
    ('LPAREN',     r'\('),
    ('RPAREN',     r'\)'),
    ('ELLIPSIS',   r'\.\.\.'),
    ('RARROW',     r'->'),
    ('QUESTIONMARK', r'\?'),
    ('INTEGER',    r'0(?![0-9])|-?[1-9][0-9]*', int),
    ('STRING', (r"""(?:"(?:[^"\n\r\\]|(?:\\u[0-9a-fA-F]{4})|(?:\\["bfnrt]))*")|""" +
                r"""(?:'(?:[^'\n\r\\]|(?:\\u[0-9a-fA-F]{4})|(?:\\['bfnrt]))*')"""),
                _str_val),
]

# Dynamically add all the token indices to globals() and __all__
__all__.extend(tok[0] for tok in _tokens)
globals().update((tok[0], i) for i, tok in enumerate(_tokens, 1))

# Regex for skipping whitespace and comments
_whitespace = r'(?:\s|(?:#.*$))*'

# Compile the token-matching and whitespace-matching regular expressions
_tokens_re = re.compile('|'.join('(' + tok[1] + ')' for tok in _tokens),
                        re.MULTILINE)
_whitespace_re = re.compile(_whitespace, re.MULTILINE)

Token = collections.namedtuple('Token', 'id, name, span, val')

def lex(ds_str):
    """A generator which lexes a datashape string into a
    sequence of tokens.

    Example
    -------

        import datashape
        s = '   -> ... A... "string" 1234 Blah _eil(# comment'
        print('lexing %r' % s)
        for tok in datashape.lexer.lex(s):
            print(tok.id, tok.name, tok.span, repr(tok.val))
    """
    pos = 0
    # Skip whitespace
    m = _whitespace_re.match(ds_str, pos)
    if m:
        pos = m.end()
    while pos < len(ds_str):
        # Try to match a token
        m = _tokens_re.match(ds_str, pos)
        if m:
            # m.lastindex gives us which group was matched, which
            # is one greater than the index into the _tokens list.
            id = m.lastindex
            tokinfo = _tokens[id - 1]
            name = tokinfo[0]
            span = m.span()
            if len(tokinfo) > 2:
                val = tokinfo[2](ds_str[span[0]:span[1]])
            else:
                val = None
            pos = m.end()
            yield Token(id, name, span, val)
        else:
            raise error.DataShapeSyntaxError(pos, '<nofile>',
                                             ds_str,
                                             'Invalid DataShape token')
        # Skip whitespace
        m = _whitespace_re.match(ds_str, pos)
        if m:
            pos = m.end()

