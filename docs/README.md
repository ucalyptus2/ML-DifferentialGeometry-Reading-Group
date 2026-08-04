# GitHub Pages site

Static single-page site generated for this reading group. Served at https://ucalyptus2.github.io/ML-DifferentialGeometry-Reading-Group/

## Regenerating `docs/tutorial/*.html` from `tutorial/*.md`

`tutorial/*.md` is the canonical source. `docs/tutorial/*.html` is generated — never
hand-edit the HTML directly, or the next regeneration silently reverts your fix.

```bash
for f in tutorial/*.md; do
  base=$(basename "$f" .md)
  [ "$base" = "README" ] && continue
  pandoc --standalone --from=gfm --to=html5 --mathjax --css=../tutorial.css \
    --include-before-body=docs/tutorial/header.html \
    -o "docs/tutorial/${base}.html" "$f"
done
for f in docs/tutorial/*.html; do
  sed -i '' -E 's/href="([0-9]{2}_[a-zA-Z_]+)\.md"/href="\1.html"/g' "$f"
done
```

Note: `tutorial/06_geometric_deep_learning_blueprint.md` links to `papers/` with the
full GitHub tree URL (not a repo-relative `../papers` link) specifically because
`docs/tutorial/../papers` doesn't exist — `papers/` is never published under `docs/`.
Keep that link absolute in the `.md` source itself; do not patch it only in the
generated HTML, since a future regeneration from `.md` will silently drop the patch.
