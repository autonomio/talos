## Change and outcome

Describe the problem, resulting behavior and scientific/user impact. Link the one open `slice` issue with `Closes #NUMBER`; also close the parent PRD only when this is its last open sub-issue. The title must match the slice title exactly.

## Validation

Record commands, results, framework lanes, distribution checks and exact candidate evidence. Identify local verification separately from CI and live protection.

## Checklist

- [ ] Review the complete diff and keep it within the slice's Surfaces and Out of Scope contract.
- [ ] Validate the promised behavior and meaningful regression coverage.
- [ ] Preserve legacy Python, SFD and CLI contracts on the shared executor.
- [ ] Verify affected framework and fresh-process archive/recovery paths with real fixtures.
- [ ] Update canonical docs and relevant public docstrings.
- [ ] Advance `talos/__init__.py`'s Hatch version and add a matching `# vMAJOR.MINOR.PATCH` changelog section.
- [ ] Update citation metadata and meet the Conventional Commits bump minimum.
- [ ] Satisfy all required checks and disposition every review thread before merge.
- [ ] Complete each slice Done Means item or record its explicit `OVERRULED` reason.

Bot exemptions apply only where declared in `governance.yml`; they do not bypass the remaining gates.
