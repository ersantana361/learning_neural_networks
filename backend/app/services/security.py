import re
from typing import Tuple


class SecurityValidator:
    BLOCKED_IMPORTS = [
        "os", "sys", "subprocess", "shutil", "pathlib",
        "socket", "urllib", "requests", "http", "httplib",
        "pickle", "marshal", "shelve", "dill",
        "builtins", "__builtins__", "importlib",
        "ctypes", "multiprocessing", "threading",
        "asyncio", "concurrent", "signal",
        "pty", "tty", "termios", "fcntl",
        "resource", "sysconfig", "platform",
        "code", "codeop", "compileall",
        "tempfile", "glob", "fnmatch",
    ]

    BLOCKED_PATTERNS = [
        (r"\bexec\s*\(", "exec() is not allowed"),
        (r"\beval\s*\(", "eval() is not allowed"),
        (r"\bcompile\s*\(", "compile() is not allowed"),
        (r"\bopen\s*\(", "open() is not allowed - use provided data instead"),
        (r"\b__import__\s*\(", "__import__() is not allowed"),
        (r"\bglobals\s*\(", "globals() is not allowed"),
        (r"\blocals\s*\(", "locals() is not allowed"),
        (r"\bgetattr\s*\(", "getattr() is not allowed"),
        (r"\bsetattr\s*\(", "setattr() is not allowed"),
        (r"\bdelattr\s*\(", "delattr() is not allowed"),
        (r"\bvars\s*\(", "vars() is not allowed"),
        (r"\bdir\s*\(", "dir() is not allowed"),
        (r"\bbreakpoint\s*\(", "breakpoint() is not allowed"),
        (r"\binput\s*\(", "input() is not allowed"),
        (r"__class__", "Accessing __class__ is not allowed"),
        (r"__bases__", "Accessing __bases__ is not allowed"),
        (r"__subclasses__", "Accessing __subclasses__ is not allowed"),
        (r"__mro__", "Accessing __mro__ is not allowed"),
        (r"__globals__", "Accessing __globals__ is not allowed"),
        (r"__code__", "Accessing __code__ is not allowed"),
        (r"__reduce__", "Accessing __reduce__ is not allowed"),
    ]

    @classmethod
    def validate(cls, code: str) -> Tuple[bool, str]:
        # Check for blocked imports
        import_pattern = r"(?:from\s+(\w+)|import\s+(\w+))"
        for match in re.finditer(import_pattern, code):
            module = match.group(1) or match.group(2)
            if module in cls.BLOCKED_IMPORTS:
                return False, f"Import of '{module}' is not allowed for security reasons"

        # Check for blocked patterns
        for pattern, message in cls.BLOCKED_PATTERNS:
            if re.search(pattern, code):
                return False, message

        return True, "Code validation passed"

    @classmethod
    def sanitize_output(cls, output: str, max_size: int = 1024 * 1024) -> str:
        # Remove ANSI escape codes
        ansi_escape = re.compile(r"\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])")
        output = ansi_escape.sub("", output)

        # Truncate if too large
        if len(output) > max_size:
            output = output[:max_size] + "\n... [Output truncated]"

        return output
