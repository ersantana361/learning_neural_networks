#!/usr/bin/env python3
import sys
import os
import json
import traceback

# Install plot interceptor before running user code
from plot_interceptor import interceptor
interceptor.install()


def execute_code():
    code = os.environ.get("CODE", "")

    if not code:
        print("Error: No code provided", file=sys.stderr)
        sys.exit(1)

    # Restricted builtins
    safe_builtins = {
        'abs': abs,
        'all': all,
        'any': any,
        'bin': bin,
        'bool': bool,
        'bytes': bytes,
        'callable': callable,
        'chr': chr,
        'complex': complex,
        'dict': dict,
        'divmod': divmod,
        'enumerate': enumerate,
        'filter': filter,
        'float': float,
        'format': format,
        'frozenset': frozenset,
        'hash': hash,
        'hex': hex,
        'int': int,
        'isinstance': isinstance,
        'issubclass': issubclass,
        'iter': iter,
        'len': len,
        'list': list,
        'map': map,
        'max': max,
        'min': min,
        'next': next,
        'object': object,
        'oct': oct,
        'ord': ord,
        'pow': pow,
        'print': print,
        'range': range,
        'repr': repr,
        'reversed': reversed,
        'round': round,
        'set': set,
        'slice': slice,
        'sorted': sorted,
        'str': str,
        'sum': sum,
        'tuple': tuple,
        'type': type,
        'zip': zip,
        '__import__': __import__,  # Needed for allowed imports
        '__name__': '__main__',
        '__doc__': None,
    }

    # Create execution namespace
    namespace = {
        '__builtins__': safe_builtins,
    }

    try:
        # Compile first to catch syntax errors
        compiled = compile(code, '<user_code>', 'exec')

        # Execute
        exec(compiled, namespace)

        # Output plots if any
        plots = interceptor.get_plots()
        if plots:
            print("\n__PLOTS__:" + json.dumps(plots))

    except SyntaxError as e:
        print(f"SyntaxError: {e.msg} (line {e.lineno})", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        tb = traceback.format_exc()
        print(tb, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    execute_code()
