# Troubleshooting

## The mini session prints different numbers than the book

Check `examples/mini-session/baseline.json` first. Every number in the book is compared
against that file, and it records the rates that produced it. If your run differs, one of
three things changed: the fixture, the rates, or your hardware. The first two are visible in
the JSON; the third is the interesting one and is what most of the book is about.

## `uv sync` resolves for a long time

That is the base environment only, and it should be quick. If it is not, you are probably
resolving a chapter's engine pin at the same time. Install those inside the chapter.

## A listing needs a GPU I do not have

Chapters 1 to 8 are written for a single 24GB card. Chapters 9 and 10 need two or more and
say so at the top. Nothing before chapter 9 requires rented hardware.

## An engine version moved and the listing broke

That is expected and it is why `appendix/` exists, dated and versioned. Check there first,
then open an issue. Engine APIs are the deliberately perishable 30 percent of this book.

## The tests in the mini session fail

They are the scoring rule for what counts as a completed task, so a failure there means the
denominator of every cost number is wrong. Read `examples/mini-session/expected.md`.
