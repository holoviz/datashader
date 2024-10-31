"""
Utilities for manipulating the Abstract Syntax Tree of Python constructs
"""
import re
import copy
import inspect
import ast
import textwrap


class NameVisitor(ast.NodeVisitor):
    """
    NodeVisitor that builds a set of all of the named identifiers in an AST
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.names = set()

    def visit_Name(self, node):
        self.names.add(node.id)

    def visit_arg(self, node):
        if hasattr(node, 'arg'):
            self.names.add(node.arg)
        elif hasattr(node, 'id'):
            self.names.add(node.id)

    def get_new_names(self, num_names):
        """
        Returns a list of new names that are not already present in the AST.

        New names will have the form _N, for N a non-negative integer. If the
        AST has no existing identifiers of this form, then the returned names
        will start at 0 ('_0', '_1', '_2'). If the AST already has identifiers
        of this form, then the names returned will not include the existing
        identifiers.

        Parameters
        ----------
        num_names: int
            The number of new names to return

        Returns
        -------
        list of str
        """
        prop_re = re.compile(r"^_(\d+)$")
        matching_names = [n for n in self.names if prop_re.match(n)]
        if matching_names:
            start_number = max([int(n[1:]) for n in matching_names]) + 1
        else:
            start_number = 0

        return ["_" + str(n) for n in
                range(start_number, start_number + num_names)]


class ExpandVarargTransformer(ast.NodeTransformer):
    """
    Node transformer that replaces the starred use of a variable in an AST
    with a collection of unstarred named variables.
    """
    def __init__(self, starred_name, expand_names, *args, **kwargs):
        """
        Parameters
        ----------
        starred_name: str
            The name of the starred variable to replace
        expand_names: list of stf
            List of the new names that should be used to replace the starred
            variable

        """
        super().__init__(*args, **kwargs)
        self.starred_name = starred_name
        self.expand_names = expand_names


class ExpandVarargTransformerStarred(ExpandVarargTransformer):
    # Python 3
    def visit_Starred(self, node):
        if node.value.id == self.starred_name:
            return [ast.Name(id=name, ctx=node.ctx) for name in
                    self.expand_names]
        else:
            return node


def function_to_ast(fn):
    """
    Get the AST representation of a function
    """
    # Get source code for function
    # Dedent is needed if this is a nested function
    fn_source = textwrap.dedent(inspect.getsource(fn))

    # Parse function source code into an AST
    fn_ast = ast.parse(fn_source)

    # # The function will be the fist element of the module body
    # fn_ast = module_ast.body[0]

    return fn_ast


def ast_to_source(ast):
    """Convert AST to source code string using the astor package"""
    import astor
    return astor.to_source(ast)


def compile_function_ast(fn_ast):
    """
    Compile function AST into a code object suitable for use in eval/exec
    """
    assert isinstance(fn_ast, ast.Module)
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)
    return compile(fn_ast, "<%s>" % fndef_ast.name, mode='exec')


def function_ast_to_function(fn_ast, stacklevel=1):
    # Validate
    assert isinstance(fn_ast, ast.Module)
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)

    # Compile AST to code object
    code = compile_function_ast(fn_ast)

    # Evaluate the function in a scope that includes the globals and
    # locals of desired frame.
    current_frame = inspect.currentframe()
    eval_frame = current_frame
    for _ in range(stacklevel):
        eval_frame = eval_frame.f_back

    eval_locals = eval_frame.f_locals
    eval_globals = eval_frame.f_globals
    del current_frame
    scope = copy.copy(eval_globals)
    scope.update(eval_locals)

    # Evaluate function in scope
    eval(code, scope)

    # Return the newly evaluated function from the scope
    return scope[fndef_ast.name]


def _build_arg(name):
    return ast.arg(arg=name)


def expand_function_ast_varargs(fn_ast, expand_number):
    """
    Given a function AST that use a variable length positional argument
    (e.g. *args), return a function that replaces the use of this argument
    with one or more fixed arguments.

    To be supported, a function must have a starred argument in the function
    signature, and it may only use this argument in starred form as the
    input to other functions.

    For example, suppose expand_number is 3 and fn_ast is an AST
    representing this function...

    def my_fn1(a, b, *args):
        print(a, b)
        other_fn(a, b, *args)

    Then this function will return the AST of a function equivalent to...

    def my_fn1(a, b, _0, _1, _2):
        print(a, b)
        other_fn(a, b, _0, _1, _2)

    If the input function uses `args` for anything other than passing it to
    other functions in starred form, an error will be raised.

    Parameters
    ----------
    fn_ast: ast.FunctionDef
    expand_number: int

    Returns
    -------
    ast.FunctionDef
    """
    assert isinstance(fn_ast, ast.Module)

    # Copy ast so we don't modify the input
    fn_ast = copy.deepcopy(fn_ast)

    # Extract function definition
    fndef_ast = fn_ast.body[0]
    assert isinstance(fndef_ast, ast.FunctionDef)

    # Get function args
    fn_args = fndef_ast.args

    # Function variable arity argument
    fn_vararg = fn_args.vararg

    # Require vararg
    if not fn_vararg:
        raise ValueError("""\
Input function AST does not have a variable length positional argument
(e.g. *args) in the function signature""")
    assert fn_vararg

    # Get vararg name
    if isinstance(fn_vararg, str):
        vararg_name = fn_vararg
    else:
        vararg_name = fn_vararg.arg

    # Compute new unique names to use in place of the variable argument
    before_name_visitor = NameVisitor()
    before_name_visitor.visit(fn_ast)
    expand_names = before_name_visitor.get_new_names(expand_number)

    # Replace use of *args in function body
    expand_transformer = ExpandVarargTransformerStarred

    new_fn_ast = expand_transformer(
        vararg_name, expand_names
    ).visit(fn_ast)

    new_fndef_ast = new_fn_ast.body[0]

    # Replace vararg with additional args in function signature
    new_fndef_ast.args.args.extend(
        [_build_arg(name=name) for name in expand_names]
    )
    new_fndef_ast.args.vararg = None

    # Run a new NameVistor an see if there were any other non-starred uses
    # of the variable length argument. If so, raise an exception
    after_name_visitor = NameVisitor()
    after_name_visitor.visit(new_fn_ast)
    if vararg_name in after_name_visitor.names:
        raise ValueError("""\
The variable length positional argument {n} is used in an unsupported context
""".format(n=vararg_name))

    # Remove decorators if present to avoid recursion
    fndef_ast.decorator_list = []

    # Add missing source code locations
    ast.fix_missing_locations(new_fn_ast)

    # Return result
    return new_fn_ast


def expand_varargs(expand_number):
    """
    Decorator to expand the variable length (starred) argument in a function
    signature with a fixed number of arguments.

    Parameters
    ----------
    expand_number: int
        The number of fixed arguments that should replace the variable length
        argument

    Returns
    -------
    function
        Decorator Function
    """
    if not isinstance(expand_number, int) or expand_number < 0:
        raise ValueError("expand_number must be a non-negative integer")

    def _expand_varargs(fn):
        fn_ast = function_to_ast(fn)
        fn_expanded_ast = expand_function_ast_varargs(fn_ast, expand_number)
        return function_ast_to_function(fn_expanded_ast, stacklevel=2)
    return _expand_varargs
