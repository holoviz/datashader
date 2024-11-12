"""
Test the DataShape lexer.
"""


import unittest

from datashader import datashape
from datashader.datashape import lexer


class TestDataShapeLexer(unittest.TestCase):

    def check_isolated_token(self, ds_str, tname, val=None):
        # The token name should be a property in parser
        tid = getattr(lexer, tname)
        # Lexing should produce a single token matching the specification
        self.assertEqual(list(lexer.lex(ds_str)),
                         [lexer.Token(tid, tname, (0, len(ds_str)), val)])

    def check_failing_token(self, ds_str):
        # Creating the lexer will fail, because the error is
        # in the first token.
        self.assertRaises(datashape.DataShapeSyntaxError, list, lexer.lex(ds_str))

    def test_isolated_tokens(self):
        self.check_isolated_token('testing', 'NAME_LOWER', 'testing')
        self.check_isolated_token('Testing', 'NAME_UPPER', 'Testing')
        self.check_isolated_token('_testing', 'NAME_OTHER', '_testing')
        self.check_isolated_token('*', 'ASTERISK')
        self.check_isolated_token(',', 'COMMA')
        self.check_isolated_token('=', 'EQUAL')
        self.check_isolated_token(':', 'COLON')
        self.check_isolated_token('[', 'LBRACKET')
        self.check_isolated_token(']', 'RBRACKET')
        self.check_isolated_token('{', 'LBRACE')
        self.check_isolated_token('}', 'RBRACE')
        self.check_isolated_token('(', 'LPAREN')
        self.check_isolated_token(')', 'RPAREN')
        self.check_isolated_token('...', 'ELLIPSIS')
        self.check_isolated_token('->', 'RARROW')
        self.check_isolated_token('?', 'QUESTIONMARK')
        self.check_isolated_token('32102', 'INTEGER', 32102)
        self.check_isolated_token('->', 'RARROW')
        self.check_isolated_token('"testing"', 'STRING', 'testing')
        self.check_isolated_token("'testing'", 'STRING', 'testing')

    def test_integer(self):
        # Digits
        self.check_isolated_token('0', 'INTEGER', 0)
        self.check_isolated_token('1', 'INTEGER', 1)
        self.check_isolated_token('2', 'INTEGER', 2)
        self.check_isolated_token('3', 'INTEGER', 3)
        self.check_isolated_token('4', 'INTEGER', 4)
        self.check_isolated_token('5', 'INTEGER', 5)
        self.check_isolated_token('6', 'INTEGER', 6)
        self.check_isolated_token('7', 'INTEGER', 7)
        self.check_isolated_token('8', 'INTEGER', 8)
        self.check_isolated_token('9', 'INTEGER', 9)
        # Various-sized numbers
        self.check_isolated_token('10', 'INTEGER', 10)
        self.check_isolated_token('102', 'INTEGER', 102)
        self.check_isolated_token('1024', 'INTEGER', 1024)
        self.check_isolated_token('10246', 'INTEGER', 10246)
        self.check_isolated_token('102468', 'INTEGER', 102468)
        self.check_isolated_token('1024683', 'INTEGER', 1024683)
        self.check_isolated_token('10246835', 'INTEGER', 10246835)
        self.check_isolated_token('102468357', 'INTEGER', 102468357)
        self.check_isolated_token('1024683579', 'INTEGER', 1024683579)
        # Leading zeros are not allowed
        self.check_failing_token('00')
        self.check_failing_token('01')
        self.check_failing_token('090')

    def test_string(self):
        # Trivial strings
        self.check_isolated_token('""', 'STRING', '')
        self.check_isolated_token("''", 'STRING', '')
        self.check_isolated_token('"test"', 'STRING', 'test')
        self.check_isolated_token("'test'", 'STRING', 'test')
        # Valid escaped characters
        self.check_isolated_token(r'"\"\b\f\n\r\t\ub155"', 'STRING',
                                  '"\b\f\n\r\t\ub155')
        self.check_isolated_token(r"'\'\b\f\n\r\t\ub155'", 'STRING',
                                  "'\b\f\n\r\t\ub155")
        # A sampling of invalid escaped characters
        self.check_failing_token(r'''"\'"''')
        self.check_failing_token(r"""'\"'""")
        self.check_failing_token(r"'\a'")
        self.check_failing_token(r"'\s'")
        self.check_failing_token(r"'\R'")
        self.check_failing_token(r"'\N'")
        self.check_failing_token(r"'\U'")
        self.check_failing_token(r"'\u123g'")
        self.check_failing_token(r"'\u123'")
        # Some unescaped and escapted unicode characters
        self.check_isolated_token('"\uc548\ub155 \\uc548\\ub155"', 'STRING',
                                  '\uc548\ub155 \uc548\ub155')

    def test_failing_tokens(self):
        self.check_failing_token('~')
        self.check_failing_token('`')
        self.check_failing_token('@')
        self.check_failing_token('$')
        self.check_failing_token('%')
        self.check_failing_token('^')
        self.check_failing_token('&')
        self.check_failing_token('-')
        self.check_failing_token('+')
        self.check_failing_token(';')
        self.check_failing_token('<')
        self.check_failing_token('>')
        self.check_failing_token('.')
        self.check_failing_token('..')
        self.check_failing_token('/')
        self.check_failing_token('|')
        self.check_failing_token('\\')

    def test_whitespace(self):
        expected_idval = [(lexer.COLON, None),
                          (lexer.STRING, 'a'),
                          (lexer.INTEGER, 12345),
                          (lexer.RARROW, None),
                          (lexer.EQUAL, None),
                          (lexer.ASTERISK, None),
                          (lexer.NAME_OTHER, '_b')]
        # With minimal whitespace
        toks = list(lexer.lex(':"a"12345->=*_b'))
        self.assertEqual([(tok.id, tok.val) for tok in toks], expected_idval)
        # With spaces
        toks = list(lexer.lex(' : "a" 12345 -> = * _b '))
        self.assertEqual([(tok.id, tok.val) for tok in toks], expected_idval)
        # With tabs
        toks = list(lexer.lex('\t:\t"a"\t12345\t->\t=\t*\t_b\t'))
        self.assertEqual([(tok.id, tok.val) for tok in toks], expected_idval)
        # With newlines
        toks = list(lexer.lex('\n:\n"a"\n12345\n->\n=\n*\n_b\n'))
        self.assertEqual([(tok.id, tok.val) for tok in toks], expected_idval)
        # With spaces, tabs, newlines and comments
        toks = list(lexer.lex('# comment\n' +
                               ': # X\n' +
                               ' "a" # "b"\t\n' +
                               '\t12345\n\n' +
                               '->\n' +
                               '=\n' +
                               '*\n' +
                               '_b # comment\n' +
                               ' \t # end'))
        self.assertEqual([(tok.id, tok.val) for tok in toks], expected_idval)
