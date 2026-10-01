# Writing docstrings

A docstring earns its place by stating a constraint, unit, return meaning, side effect or failure boundary that the signature does not express. The adopted gates check form and measured debt; reviewers check meaning.

## Prerequisites

Read the function and its callers. Use the current signature as the source of defaults and types. Run `python governance/check_docstrings.py` and `python governance/check_module_docstrings.py` after changing the affected source.

## Function and method rules

| Rule | Reason |
| --- | --- |
| Begin with the delivered result; avoid `calculate`, `generate`, `make` and `build` as title verbs | Describe what the caller receives |
| Do not repeat default values in description prose | The signature owns the default |
| Write a note marker as `NOTE:` | Keep explicit notes searchable |

Suitable verbs include Return, Parse, Validate, Resolve and Extract. A well-formed sentence that repeats the symbol's name still adds no information. For Talos, document metric direction, array shape, trial ordering, persistence side effects and recovery boundaries when those are material.

## Module rules

A non-empty module carries a concise purpose statement unless the configured exemption applies. Historical debt is measured; do not remove an exemption or change the scan surface to hide a failure. Do not fabricate meaningless module docstrings to reduce a count.

Strict punctuation and formatting checks come from the Ruff profile; measured quality enforcement and the docstring scanners must agree with the declared baseline. New violations cannot be buried among existing ones.

## Read next

- [Documentation system](Documentation-System.md)
- [Configuration](Configuration.md)
- [Developer home](README.md)
