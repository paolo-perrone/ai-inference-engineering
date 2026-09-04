# Setup

Python 3.11 or newer. Chapters 1 to 8 run on a single 24GB GPU. Chapters 9 and 10 need two
or more real cards, which is the only place in the book that costs money.

```bash
git clone https://github.com/paolo-perrone/ai-inference-engineering
cd ai-inference-engineering
uv sync
python3 examples/mini-session/measure.py --self-test
```

That last line should print PASS. If it does not, nothing else in the repo will behave, and
that is a bug here rather than in your setup.

## Per-chapter environments

The base install carries only what every chapter needs. Anything engine-specific is pinned
inside the chapter that uses it, because engine versions move faster than anything else in
this book and a shared lockfile would make every reader resolve all of them.

## What to do when a listing stops running

Open an issue. A listing that no longer runs because something moved underneath it is a bug
in this repository, not in your machine. That is the promise the book's durability argument
rests on, and it is the repo's job rather than the printed page's.
