# Updating this public selection

Research continues in the private checkout. Treat this repository as a curated
export, never as a mirror. `PUBLICATION.json` records the source revision,
each selected path, its origin, its SHA-256 and the figure recipe.

1. Choose an immutable research commit. In a separate temporary clone, check
   out that commit detached. Do not switch, move or reconfigure the working
   research checkout.
2. Run the private `publication/export_public.py` with `--source` pointing to
   that clean clone and `--destination` pointing to a new empty directory
   outside it. Its `EXACT` and `TEMPLATES` lists are the explicit allowlist.
   Review additions individually. The exporter does not copy whole folders.
3. Review attribution, dataset permission, limitations and both extracted
   aggregate tables. Update portfolio-authored text when findings change;
   never imply that a new source snapshot reran the experiments.
4. Clone the public repository separately, create a publication branch, and
   copy only allowlisted release files into it. Compare the previous public
   `PUBLICATION.json` to catch removals explicitly; delete only a specifically
   reviewed obsolete public file. Never copy `.git`, hidden tool settings,
   private logs, dissertation, source datasets or full results.
5. Install the pinned environment in a new venv, run the README commands in
   a clean clone of the prepared public commit, and regenerate both figures.
   Inspect staged paths, content, all outgoing commits, file sizes, source
   hashes and secret-scan findings. Commit results of the review privately.
6. Push only the reviewed public branch/commit with an ordinary push. Preserve
   existing public ancestry; never merge private branches, push private refs,
   force push, or change private repository visibility.
7. Record the full research source SHA ↔ public commit SHA, URL, explicit
   file manifest and validation in private `publication/releases/` and update
   the private project's state/log. Public-origin development should return
   to the private workflow as explicitly reviewed changes.

For this first edition, both older repositories were private. The existing
`QuantumPhases` repository was left intact and a new public repository was
created with independent history. Later updates must preserve this public
repository's own history.
