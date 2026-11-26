# BeaverTails attack configs

Config names encode the response format and non-default attack settings:

- `nl` and `sp` select newline and space response templates.
- AdamW is the default optimizer and is omitted from ordinary attack names.
- `adamw-no-history` sets both Adam moment coefficients to zero.
- `ordered` disables dataset shuffling.
- `sgd`, `clipped`, and `cosim` identify analysis variants.

The RepNoise configs form a complete format/optimizer matrix. SEAM configs
without an analysis suffix are the current shuffled attacks; the remaining
files reproduce the ordered, optimizer, and cosine-tracking comparisons.
