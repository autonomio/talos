# Documentation system contract

This page defines the maintained Talos documentation system. It governs source
ownership, page composition, routes, rendering and verification. The repository
root README is preserved during this adoption; its next revision is a separate
step. Runtime behavior remains governed by implementation and acceptance tests.

`MUST`, `SHOULD`, and `MAY` identify required, recommended, and optional parts
of the contract.

## Prerequisites

- Node.js 20.18.1 or later
- npm dependencies installed from `docs-site/package-lock.json` with `npm ci`
- the runtime required to prove product-specific examples
- a canonical repository URL, documentation origin, and base path

## Quality bar

Documentation is release-ready only when it is:

- correct against current source, exports, schemas, workflows, and behavior
- complete across the declared maintained Markdown corpus
- coherent from product entry through guides, reference, maintenance, and
  package boundaries
- runnable in the dependency environment each example declares
- consistent in terminology, units, page roles, links, composition, and visual
  treatment
- accessible across keyboard, desktop, mobile, light, and dark contexts
- secure and discoverable through deployment headers, canonical metadata,
  sitemap, and robots behavior
- mechanically protected in proportion to each claim

## Portability boundary

The system has three layers.

| Layer | Owns | Must not own |
|---|---|---|
| Shared scaffold | five-section architecture, page contracts, source-to-route invariants, rendering, visual tokens, interaction behavior, generic gates | product identity, claims, examples, routes, package inventory, or deployment account |
| Product profile | identity, tagline, repository URL, site origin, base path, source inventory, routes, product narrative, dependencies, risk boundary | copies of shared rendering logic or weakened acceptance rules |
| Deployment adapter | hosting configuration, redirects, headers, preview and rollback | documentation content or alternate route authority |

The portable implementation is `docs-site/` plus the starter Markdown corpus.
Product-specific values enter only through `product-docs.json`, `docs-map.json`,
and maintained source pages. Hosting remains repository-specific because
Cloudflare, GitHub Pages, object storage, and internal platforms expose
different controls.

The scaffold is portable only when:

- its profile schema expresses every product-specific value
- shared files contain no product claim or route literal
- every maintained source maps to one stable route
- a repository can update shared files without replacing its profile or prose
- required checks cannot be disabled while claiming the same standard
- one build path remains after migration and rollback expiry

## Adoption baseline

The documentation scaffold is pinned to commit
`fa7bf92caab5bb8d7f73120d4dc3526fd86c0348`. The Talos content baseline is
`c0db9eb0404b8b8f1181f0b8069f68fc45fffca5`. Existing walkthroughs and
references remain canonical sources; the original getting-started explanation
moves intact to [Quickstart](../Guides/Quickstart.md).

The visual authority for this adoption is the supplied Autonomio style guide,
not the scaffold's original theme. See [Documentation style](Documentation-Style.md)
for exact values and the accessible dark-mode adaptation. The scaffold retains
local search, source-aware edit links, metadata and automated proof.

## Source ownership

- [README.md](../../README.md) is the product home and first-success path.
- [docs/README.md](../README.md) is the public task router.
- Guide sources, including established flat files in `docs/`, own end-to-end
  jobs and operational workflows. `docs/Guides/README.md` routes these tasks.
- Reference sources, including established flat files in `docs/`, own interfaces,
  schemas, defaults, outputs, and edge cases. `docs/Reference/README.md` routes them.
- `docs/Developer` owns contributor and maintainer guidance.
- package `README.md` files own package boundaries and public entry points.
- `docs-site/docs-map.json` is the complete maintained source and route map.
- `.generated`, `.docusaurus`, and `build` are derived and MUST NOT be
  hand-authored or committed.

Author a claim once. Secondary pages summarize and link to its canonical home.

## Information architecture

The site presents five top-level sections.

| Section | Responsibility |
|---|---|
| Overview | product boundary, system story, task routing, and roadmap |
| Guides | end-to-end jobs and operational workflows |
| Reference | interfaces, schemas, defaults, outputs, and edge cases |
| Developer | contribution, documentation, release, security, and maintenance |
| Packages | module ownership, public entry points, and package boundaries |

Top-level categories start collapsed. Every maintained source MUST map to
exactly one destination and stable public route. Every route MUST have one
primary page role. Generated category indexes are navigation surfaces, not
alternate canonical explanations.

## Product narrative

Every repository defines one source-backed sequence from input to observable
outcome. It belongs in `docs/README.md` and guides reader order across pages.

The sequence MUST:

1. start at the real user entry point
2. name implemented boundaries in execution order
3. show the first observable result
4. state where product responsibility ends
5. link each step to its canonical guide or reference

The shared scaffold requires this sequence but never supplies product steps.

## Page composition

Composition is the required order of information, not a demand for identical
headings. Omit an inapplicable element only when the omission is explicit.

### Product home

1. product identity and one-sentence value
2. owned and excluded product boundary
3. current capabilities
4. minimum install and first successful workflow
5. observable outputs or artifacts
6. risk boundary
7. routes by reader task
8. contribution, support, security, citation, and license

### Docs hub

1. product in one page
2. start routes grouped by reader job
3. product narrative sequence
4. complete top-level docs map
5. owned and excluded product boundary
6. explicit next routes

### Guide

1. job, outcome, and current scope
2. prerequisites or an explicit statement that none are required
3. ordered procedure with a concrete command or example
4. expected output, artifact, or observable result
5. failure boundaries and material edge cases
6. next task

### Reference

1. covered surface and abstraction boundary
2. public names and import pattern
3. signatures, parameters, defaults, returns, units, and side effects
4. minimum concrete example
5. edge cases, errors, and optional dependencies
6. relationship to adjacent surfaces
7. next reference or workflow

### Developer page

1. purpose and authority
2. prerequisites
3. scope and ownership boundaries
4. executable process or checklist
5. required proof and failure conditions
6. review or maintenance boundary
7. related maintainer routes

### Package README

1. package path and one-sentence responsibility
2. canonical public docs
3. owned and excluded boundaries
4. source-true public entry points
5. adjacent packages and optional dependencies
6. compact source-tree orientation when it materially helps
7. operational caveats
8. next routes

## Visual system

`docs-site/src/css/custom.css` is the visual authority. Product identity changes
content and metadata; it does not fork the shared component rules.

The supplied Autonomio style guide governs Finlandica Regular, palette,
spacing, text links and responsive navigation. The implementation lives in
`docs-site/src/css/custom.css`; the exact applied values and page-role adaptation
are maintained in [Documentation style](Documentation-Style.md).

The reading column is bounded to 680px inside the guide's centered 1344px
wrapper. Article pages preserve their established content order. Generated
category indexes use full-row links with rules rather than cards. The design
adopts the guide's typography and geometry without introducing its unrelated
research-site page inventory.

### Interaction contract

- Navbar order is Home, Overview, Guides, Reference, Developer, Packages,
  then GitHub.
- Light, dark, and system theme choices are supported.
- Search is local, keyboard reachable, and links to a complete search page.
- Top-level categories start collapsed; the current section expands.
- The desktop sidebar is hideable.
- Mobile keeps the same text navigation visible below the wordmark, without a
  hamburger menu. On-page contents remain collapsible.
- Breadcrumbs, anchors, pagination, copy controls, and skip-to-content remain.
- Every edit link targets its real source under repository `/edit/master/`, configured by `sourceBranch`.
- Blog and standalone pages are disabled; documentation owns the root route.

## Writing rules

- Lead with current behavior and reader impact.
- Give each page one primary role and one primary audience task.
- Prefer exact commands, paths, values, units, signatures, return fields, and
  observable results.
- Use American English for shared prose.
- Use `python` fences only for standalone parseable code.
- State required extras before an example imports an optional dependency.
- Do not present local measurements as stable API guarantees.
- Do not describe planned, historical, or external behavior as current.
- Use relative links between maintained sources.
- End task-oriented pages with an explicit next route.

## Source-backed claims

Use the narrowest authoritative source.

| Claim | Authority |
|---|---|
| import or export | package `__init__.py` and an import smoke test |
| callable arguments or defaults | current function or class signature |
| schema or configuration field | parser, validator, and shipped example |
| result field or artifact | implementation and focused test |
| dependency | project manifest and lockfile |
| release behavior | current workflow and script |
| hosted route | `docs-map.json`, assembler, and static build |
| visual value | `docs-site/src/css/custom.css` |
| deployment behavior or header | deployment adapter and live HTTP response |

When prose and source disagree, fix the prose or route a separate behavior
defect. Documentation work MUST NOT silently change runtime contracts.

## Examples

Examples satisfy the level they imply:

- syntax examples parse
- import examples import in the declared environment
- command examples use current argument order and names
- runnable workflows complete against a bounded fixture
- output examples contain only fields the implementation produces
- destructive, costly, networked, or long-running effects are stated first

## Links and route policy

- Local links and fragments resolve in source and assembled output.
- Public links use the canonical route shape without optional trailing slash.
- Every page receives a source-true `custom_edit_url`.
- Repository files outside the map may link to GitHub.
- Published routes are stable API.
- A changed route requires redirect, link migration, sitemap update, and an
  explicit expected difference.
- External links pass the automated status check; initial and redirected
  destinations must resolve only to public network addresses.

## Assembly and configuration

| Surface | Responsibility |
|---|---|
| `product-docs.json` | product identity and deployment coordinates |
| `docs-map.json` | five sections, maintained sources, destinations, and routes |
| `assemble-docs.mjs` | validation, front matter, link rewriting, MDX normalization, categories, robots |
| `docusaurus.config.js` | metadata, search, navigation, footer, theme loading |
| `sidebars.js` | generated navigation |
| `src/css/custom.css` | visual tokens, typography, components, responsive behavior |
| `package.json` and lockfile | commands and deterministic dependency graph |
| `verify-build.mjs` | route, sitemap, robots, search-index, and asset proof |
| `tests/docs.spec.js` | rendered interaction, layout, theme, and accessibility proof |
| `.github/workflows/ci.yml` | documentation security, full site checks, executable Python examples, framework acceptance and retained proof |

The assembler:

1. validates the product profile and five-section route map
2. recreates the ignored generated directory
3. maps every maintained source to one destination and route
4. writes source-aware front matter and edit URLs
5. rewrites relative links between mapped sources
6. preserves GitHub links for repository files outside the corpus
7. normalizes source Markdown for MDX
8. creates collapsed category metadata and robots output

## Delivery, security, and discovery

The static build guarantees:

- one canonical origin and normalized base path
- Open Graph and Twitter summary metadata from the product profile
- a sitemap containing every page and generated category index
- a robots resource naming the canonical sitemap
- local search without an external indexing service
- no known production dependency advisory at check time

The deployment adapter MUST additionally prove canonical redirects, TLS,
content security, permissions, referrer, content-type, frame headers, and `404`
behavior. These controls cannot be portable without choosing a host.

## Verification matrix

| Concern | Mechanism |
|---|---|
| profile and one-to-one routing | assembler validation |
| local links | Docusaurus broken-link failure |
| external links | `check-external-links.mjs` |
| Markdown style | `markdownlint-cli2` |
| dependency advisories | `npm audit --omit=dev` |
| assembly and production build | Docusaurus build |
| routes, sitemap, robots, and search | `verify-build.mjs` |
| static asset budgets | `verify-build.mjs` |
| desktop/mobile and light/dark rendering | Playwright |
| local search, navigation, and edit links | Playwright |
| keyboard and WCAG 2 A/AA accessibility | Playwright and Axe |
| source-linked product semantics | repository-specific tests |
| deployment redirects and headers | repository-specific HTTP tests |
| representative visual comparison | parity evidence on visual changes |

## Required commands

From the repository root:

```bash
npm --prefix docs-site ci
npm --prefix docs-site run security:audit
npm --prefix docs-site run check
```

The documentation workflow installs locked dependencies and runs the same
security audit and full check. Python example execution remains in the existing
CI workflow; site rendering is not evidence that model code ran.

For changed product examples, also run their repository-specific semantic
tests. For layout, navigation, search, theme, route, or shared-system changes,
execute the parity protocol.

## Parity protocol

Parity preserves intended behavior, composition, and appearance. It does not
preserve documented defects, security weaknesses, broken routes, or
inaccessible behavior.

### Freeze the baseline

Record:

- baseline commit SHA and deployed URL
- lockfiles and tool versions
- maintained source inventory and route map
- sitemap URL set and status
- navigation, search, edit-link, and theme behavior
- CSS tokens and computed geometry
- headers, redirects, metadata, and error responses
- representative screenshots
- known defects and their candidate disposition

Retain the previous production artifact for a future publishing cutover.
For this local adoption, retain baseline source hashes, content inventory and
known Docsify defects in the adoption receipt. Build the candidate with the
required commands above. Publishing is a separate operation.

### Compare invariants

Require equality unless the slice or PR declares an expected difference:

- maintained sources, mappings, and route set
- navbar, categories, sidebar, footer, pagination, and edit targets
- search availability and representative results
- metadata, sitemap, headers, and status codes
- shared tokens, geometry, responsive breakpoint, and theme behavior
- all non-target content and representative screenshots

The minimum visual matrix covers product home, one category index, guide,
reference, developer page, and package page at:

- desktop `1440x900`, light and dark
- mobile `390x844`, light and dark

Every expected difference names the route, viewport, theme, old and new
behavior, source of authority, evidence, and rollback effect. Unlisted
differences fail parity.

### Accept and retain rollback

Acceptance requires:

- all required commands green on the exact candidate
- zero unlisted route, interaction, metadata, console, or visual differences
- known defects fixed or linked; none silently accepted as parity
- candidate preview reviewed before cutover
- previous production artifact retained until smoke checks pass

## Adoption protocol

### Talos adoption

1. Freeze the current sources, code blocks, routes and known renderer defects.
2. Adopt the documentation scaffold and this contract only.
3. Map the maintained corpus into the five sections without discarding the
   existing teaching content.
4. Set Talos identity, Autonomio ownership, repository coordinates and routes.
5. Run locked installation, the security audit, the full site check and changed
   product-semantic checks.
6. Review desktop/mobile and light/dark previews, routes and content retention.
7. Replace the Docsify source entry point with one documentation build path.
   Retain the deployed production artifact until a separately authorized
   publication has passed live smoke checks.

The rest of the repository template is outside this adoption. Root README
revision is also deferred. Neither exclusion weakens the route, accessibility,
content-retention or example-execution checks applied here.

## Versioning and drift control

- Patch: compatible defect, gate repair, or token correction.
- Minor: backward-compatible component, page contract, gate, or profile field.
- Major: incompatible profile schema, architecture, route policy, token
  contract, or required gate.

Shared changes document supported runtimes, profile compatibility, changed
contracts, migration, rollback, and parity evidence. A local override requires
an owner, reason, issue, expiry, and proof that accessibility, security, route,
and quality behavior remains intact.

## Review checklist

- Does each claim have a current authoritative source?
- Does each page keep one role and its required composition?
- Are product values confined to profile, route map, and content?
- Are prerequisites, outputs, failures, and next routes explicit?
- Is every maintained source assembled exactly once?
- Do edit links target real source files?
- Are local and external links healthy?
- Do desktop/mobile and light/dark states match the visual contract?
- Are keyboard, focus, contrast, landmarks, tables, and code usable?
- Are metadata, sitemap, robots, redirects, headers, and errors correct?
- Do semantic tests, lint, audit, build, and browser checks pass?
- Is every expected delta declared and every unexpected delta resolved?
- Is rollback available through post-cutover smoke checks?

## Read next

- [Documentation hub](../README.md)
- [Documentation-site operation](../../docs-site/README.md)
- [Interface reference](../Reference/README.md)
- [Developer home](README.md)

## Public documentation and coverage

Successful protected-master runs of the Test and build workflow retain the verified static site. The Pages publisher deploys that artifact after the complete run succeeds and skips superseded master assessments. Pull-request code is never executed by the publisher. The deployed statement coverage badge links its complete-package report, source commit and originating CI run; the initial static placeholder makes no measured claim. Repository-owned documentation links resolve against mapped source pages or checked-in static assets before deployment, while external destinations still require successful HTTP responses.

The Pages publishing source must use GitHub Actions. [Setup](../../SETUP.md) owns repository activation; building the site locally does not establish publication.

## Bounded executable verification

The documentation runner applies one 30-minute deadline to all queued and
running components. Expiry kills the component's entire process group and
records a failed executor; it cannot become successful documentation evidence.
The host job also has a 45-minute total timeout, including installation and
framework acceptance.

CI executes the full framework suite once. The identical contributor test
command may reuse that successful execution only when every tracked source
file, the interpreter, every locked documentation dependency and the CPU
runtime environment match. The receipt also binds the workflow run and
attempt. Root reports retain the original proof and mark the reused command;
other development commands execute normally. Standalone local verification
executes the contributor suite unless an explicit matching receipt is supplied.
Identical site command fences reuse one successful local execution only while
all tracked inputs remain unchanged. Reports distinguish execution from reuse.
All fences, notebook cells, model examples, archive and recovery checks remain
in the inventory; reuse neither removes them nor invents execution counts.
