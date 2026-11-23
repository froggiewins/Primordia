# Primordia

A 3D evolution game in a single HTML file: grow from a cell in the tide pools into a land creature, then lead a tribe. Build your creature in a drag-and-drop editor with a bendable spine and 62 unlockable parts, each with its own abilities.

**Play:** https://froggiewins.github.io/Primordia/

## Files
- `primordia.html`: the playable game, with all models embedded.
- `src/primordia.src.html`: the game source. Edit this one, not `primordia.html`.
- `models/build_models.py`: builds every part model in Blender and exports `primordia_parts.glb`.
- `tools/embed_models.py`: embeds the GLB into the source and writes `primordia.html`.

## Rebuild
```
"C:\Program Files\Blender Foundation\Blender 4.4\blender.exe" -b --python models\build_models.py
python tools/embed_models.py
```
