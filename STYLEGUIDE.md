# Python Style Guide

## Table of Contents

- [Introduction](#introduction)
- [Code Formatting](#code-formatting)
- [Naming Conventions](#naming-conventions)
- [Imports](#imports)
- [Comments and Docstrings](#comments-and-docstrings)
- [Error Handling](#error-handling)
- [Best Practices](#best-practices)

## Introduction

This document outlines the coding style and best practices for writing Python code in this project.

## Code Formatting

- Follow [PEP 8](https://peps.python.org/pep-0008/).
- Use 4 spaces per indentation level, no tabs.
- Keep line length to a maximum of 79 characters.
- Use blank lines to separate functions and class definitions.
- Use meaningful spacing inside parentheses and around operators.

### Auto Code Formatting
For auto code formatting, install [Ruff](https://marketplace.visualstudio.com/items?itemName=charliermarsh.ruff).
After this add the following lines to `settings.json`:
```json
{
  "[python]": {
    "editor.formatOnSave": true,
    "editor.codeActionsOnSave": {
      "source.fixAll": "explicit",
      "source.organizeImports": "explicit"
    },
    "editor.defaultFormatter": "charliermarsh.ruff"
  }
}
```
This will make it so that [PEP 8](https://peps.python.org/pep-0008/) is auto applied when saving the file.

## Naming Conventions

- Use `snake_case` for variable and function names.
- Use `PascalCase` for class names.
- Use `UPPER_CASE` for constants.
- Prefix unused variables with an underscore (`_`).

## Imports

- Use absolute imports where possible.
- Group imports in the following order:
  1. Standard library imports
  2. Third-party library imports
  3. Local application imports
- Use one import per line.

## Comments and Docstrings

- Use comments to explain why a piece of code exists, not what it does.
- Try to write code that can explain itself, so no comments are needed.
- Use `#` for single-line comments and place them above the code they reference.
- Write docstrings for all public functions, classes, and modules.

Example:

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Briefly explain what this function does.

    Params:
    -------
    - param1 (int): Description of param1.
    - param2 (str): Description of param2.

    Returns:
    ---------
    - bool: Description of the return value.
    """
    return True
```

## Error Handling

- Use exceptions for error handling, not return codes.
- Catch specific exceptions instead of using a bare `except:`.
- Use `try/except/else/finally/asserts` blocks where appropriate.
- Before a function returns a value, check if this is the aspected value.

Examples:

```python
try:
    value = int(user_input)
except ValueError:
    print("Invalid input. Please enter a number.")
```

```python
def example_function(param1: int, param2: str) -> bool:
    """
    Briefly explain what this function does.

    Params:
    -------
    - param1 (int): Description of param1.
    - param2 (str): Description of param2.

    Returns:
    ---------
    - bool: Description of the return value.
    - Image is created, if it writes something to the machine
    """
    foo = True
    assert isinstance(foo, bool) 
    return True
```

## Best Practices

- Keep functions small and focused on a single task.
- Avoid using global variables.
- Use list comprehensions where appropriate.
- Prefer f-strings for string formatting.
- Write unit tests for critical code paths.


---
