# Writing rules for this paper

This directory is a research paper draft. These rules exist because
LLM-drafted prose has a recognizable fingerprint, and reviewers, and
anyone skimming for competence, notice it fast. The rules below are
compiled from what people (editors, ML researchers, academic writing
guides) actually flag as "AI slop" in technical/academic writing
specifically, not general blog-post advice. Apply them to every
`.tex` file in `sections/` before considering a draft done.

Sources consulted: [SlopDetector — Signs of AI Writing](https://slopdetector.org/blog/signs-of-ai-writing),
[Wikipedia — Signs of AI writing](https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing),
[Paperpal — Reasons your writing looks AI-like](https://paperpal.com/blog/academic-writing-guides/reasons-your-writing-looks-like-ai-and-how-to-fix-it-manually),
[Walter Writes — ChatGPT words to avoid](https://walterwrites.ai/most-common-chatgpt-words-to-avoid/),
[robotsatemyhomework — AI writing patterns](https://robotsatemyhomework.substack.com/p/ai-writing-patterns),
[McGill OSS — Why did LLMs steal our em-dashes](https://www.mcgill.ca/oss/article/critical-thinking-student-contributors-technology/why-did-llms-steal-our-em-dashes),
[conorbronsdon/avoid-ai-writing SKILL.md](https://github.com/conorbronsdon/avoid-ai-writing/blob/main/SKILL.md),
[Blake Stockton — Colons, Colons Everywhere](https://www.blakestockton.com/colons-everywhere/).

## 1. Banned punctuation habit: the em dash

The single strongest tell in a technical draft. One editor's line,
worth keeping in mind while writing: "when I see three em dashes on
the same page of a methods section, I don't need a detection tool."
LLM output uses them as connective filler where a human would commit
to a period, a comma, a colon, or a parenthetical instead.

- Default to a period and a new sentence.
- Use a colon only when what follows is genuinely a restatement,
  list, or explanation of what precedes it, not as a generic
  connector.
- Use parentheses for a true aside, not as a hedge.
- If you catch yourself writing `---` in the `.tex` source, stop and
  pick one of the above. A hard rule: at most one em dash per
  paragraph, and only where no other punctuation does the job.

## 2. Banned vocabulary

Do not use these words. They are true tells: an LLM's default
register reaches for them at a rate no domain expert writing
naturally does.

`delve, leverage, navigate, elevate, intricate, meticulous(ly),
synergy, empower, landscape, ecosystem, underscore(s), seamless,
robust, game-changer, harness (v.), streamline, pivotal, innovative,
cutting-edge, paradigm, groundbreaking, revolutionary, transformative,
unlock, unleash, beacon, endeavor, undoubtedly, testament, tapestry,
underpinning(s), garnered, encompassing, burgeoning, realm, holistic,
comprehensive (as a filler adjective), crucial, key (as a filler
adjective), significant/significantly (unless reporting an actual
statistical test)`.

If a sentence needs one of these to make its point, the sentence is
underspecified. Say the specific thing instead: not "this underscores
the importance of X," but the actual reason X matters, stated plainly.

## 3. Banned transition words

`furthermore, moreover, consequently, notably, importantly,
additionally` used as paragraph openers. These are the words a model
reaches for to glue unrelated sentences together without doing the
work of an actual logical connective. If two sentences are actually
related, the relationship should be inferable from their content, or
stated with a specific connective (`because`, `so`, `but`, `since`,
`which means`) that names the relationship instead of gesturing at
"there is a relationship here."

Never start two consecutive paragraphs, or two paragraphs in the same
section, with the same transition word.

## 4. Banned rhetorical patterns

- **"Not X, but Y" / "not merely X, Y instead."** Overused by LLMs as
  a cheap way to sound like it's making a sharp distinction. Earn a
  contrast by stating what's actually different, or drop the
  scaffolding and just state Y.
- **Rule of three.** Forced triads (`A, B, and C` where a real list
  would have two items or five) are a stylistic tic, not an argument.
  Only group things in threes when there really are three, and vary
  list lengths across the draft.
- **Mechanical parallelism across paragraphs.** If every paragraph in
  a section has the identical shape (topic sentence, citation dump,
  boilerplate "we differ from this by..." closer), it reads as
  templated. Vary structure, length, and how each paragraph resolves.
  A related-work paragraph does not need to end by explicitly
  positioning the paper against every single citation in it.
- **Meta-commentary about the text itself.** Avoid "in this section,
  we...", "this paper presents...", "the following describes...".
  State the content; the section heading already told the reader what
  section they're in.
- **Unsupported authority claims.** "studies show," "research
  suggests," "it is well known that" without a citation immediately
  attached. Every claim needs either a citation or to be something
  the paper itself demonstrates.
- **Hedge-then-assert padding.** "It is worth noting that," "it should
  be emphasized that," "arguably." These add words, not information.
  Delete the hedge and state the claim, or don't make the claim.
- **A conclusion/abstract that just re-narrates itself.** Don't
  restate the abstract in the introduction's closing paragraph, or the
  introduction in the conclusion. Each retelling should add
  information (a number, a mechanism, an implication) not repeat the
  same sentence with synonyms swapped in.

## 5. Formatting rules specific to LaTeX

- No bold-label bullet lists of the form `\textbf{Label:} sentence.
  \textbf{Label:} sentence.` repeated more than twice in a row unless
  it's a genuinely enumerable structure (e.g. the four design
  invariants, which are individually numbered claims, not this
  pattern).
- Don't bold key terms inline in running prose for emphasis; if a term
  is worth flagging on first use, define it once and move on. Reserve
  `\textbf{}` for genuine structural labels (paragraph headers,
  contribution items).
- Section and `\paragraph{}` headers: sentence case, not title case,
  and not a noun phrase followed by a colon on every single one
  (`\paragraph{Foo bar.}` as a short label is fine and matches this
  paper's existing convention; don't turn every paragraph into `Foo
  Bar: An Analysis of Baz`).
- Vary sentence length on purpose. A paragraph of uniformly
  medium-length sentences (the classic LLM rhythm) reads as
  machine-generated even when every individual sentence is fine. Mix
  in a short sentence after a long one, especially to land a claim.

## 6. Positive rule: specificity beats hedged generality

The actual fix underlying all of the above is the same in every case:
replace a vague, safe, general statement with the specific claim,
number, mechanism, or citation that it was standing in for. AI slop
is, structurally, a generator that is optimizing for "sounds plausible
and complete" rather than "says something falsifiable." A technical
paper's prose should read like it was written by someone who ran the
experiment and is telling you exactly what happened, not by something
trying to sound like it did.

Before finishing a section: read it once for content (is every claim
either cited or backed by something in this repo/these results?), and
once for rhythm (read it aloud; if it sounds like a press release,
rewrite it).
