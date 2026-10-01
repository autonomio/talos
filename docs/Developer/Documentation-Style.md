# Documentation style

This page applies the supplied Autonomio website style guide to Talos
technical documentation. The guide governs visual treatment and voice; the
[documentation system](Documentation-System.md) governs page roles and proof.
It does not import the guide's research-site content or change library behavior.

## Prerequisites and authority

Read the supplied two-page [Autonomio style guide](../_media/autonomio-style-guide.pdf)
before changing these values. The implementation is `docs-site/src/css/custom.css` and the navigation
theme components. Fonts are self-hosted through the locked
`@fontsource/finlandica` dependency. Code uses the system monospace stack.

## Palette

| Token | Value | Use |
| --- | --- | --- |
| Paper | `#F7F7F2` | Page and navigation background |
| Ink | `#252D33` | Primary text |
| Muted | `#626B72` | Secondary text |
| Line | `#D4DADD` | One-pixel rules |
| Wash | `#EAF0F3` | Talos section only |
| Blue | `#2A4A70` | Wordmark, links and small accents |

No gradients, shadows or additional accent colors. In dark mode, invert paper
and ink for the reading surface and use line for secondary text and links. Keep
the navigation paper with its blue wordmark. This is an accessible adaptation
within the supplied palette; the PDF specifies no separate dark palette.

## Typography and geometry

Finlandica Regular 400 carries body text, headings, navigation and the wordmark.
Headings use sentence case, left alignment and ragged-right text. Preserve
semantic emphasis in existing prose without adding bold display typography.

| Role | Desktop / mobile | Line height | Tracking |
| --- | --- | --- | --- |
| AUTONOMIO wordmark | 25 / 24px | 1.2 | +0.235em |
| Footer wordmark | 18 / 17px | 1.2 | +0.22em |
| Talos title | 91 / 74px | 1.0 | -0.045em |
| Reading title | 94 / 65px | 1.015 / 1.035 | -0.037 / -0.033em |
| Section heading | 49 / 38px | 1.12 | -0.025em |
| Article heading | 31 / 27px | 1.19 | -0.018em |
| Body | 19 / 18px | 1.70 | 0 |
| Lead | 25 / 23px | 1.46 | -0.012em |
| Navigation | 16 / 15px | 1.40 | 0 |
| Summary | 18 / 17px | 1.60 | 0 |
| Code | 14 / 13px | 1.75 | 0 |

The centered border-box wrapper has a 1344px maximum width including padding.
Gutters are `clamp(24px, 6.25vw, 96px)`: at 1440px its content width is 1164px
and its outer content margin is 138px. At 390px content is 342px wide, with
24px gutters. At 359px and below gutters become 20px. The desktop header is
142px high. Navigation moves below the mark at 520px and below.

Technical reading uses a column of at most 680px within that wrapper, beside
source navigation on desktop. The category lists follow the guide's reading
view: full-row links with one-pixel rules, 34px top and 37px bottom spacing on
desktop, without cards. Body paragraphs use 22px gaps. Article starts use the
guide's 78px desktop and 41px mobile top spacing; the closing area has 96px
desktop and 67px mobile space.

## Interaction

Keep all five sections, Home and GitHub visible as text navigation on narrow
screens; there is no hamburger menu. Search and theme controls remain keyboard
reachable. Technical tables and code may scroll horizontally within their own
containers; the page must not overflow.

Links use one-pixel underlines at a 0.24em offset. Active navigation is blue,
with a seven-pixel underline offset. Focus uses a two-pixel blue outline with a
six-pixel offset. Dark reading surfaces use the high-contrast line token for
focus. Preserve the skip link and reduced-motion support. Use text controls
rather than prominent filled buttons. No decorative animation or imagery.

## Verification and maintenance

Run the site commands from [Documentation system](Documentation-System.md).
Review product home, a category, a guide, reference, developer and package pages
at 1440×900 and 390×844 in both themes. Check wordmark geometry, regular font
weight, readable line length, all navigation links, focus, search, copied code
and horizontal overflow. Accessibility checks must pass rather than being
suppressed to match an image.

Changes to these values require updating the CSS and browser assertions together,
recording the expected visual difference and retaining reviewed screenshots.
Use the guide's quiet, exact voice: state current behavior, dates, methods and
observed results; remove claims whose evidence is absent.

## Read next

[Site operation](../../docs-site/README.md) describes the build and preview.
[Documentation system](Documentation-System.md) defines the acceptance boundary.
