# AnyTop-13 Minipack to Planet Zoo Mesh

This is the shortest supported route for visualising a generated PZ motion on
its original weighted Planet Zoo mesh. It is for the private 311-species
`Tevior/anytop13-animal-minipack` and the private skinning resource package.

## Inputs

For one `PZ_*` object, obtain:

- generated raw (de-normalised) `motion_13ch.npy`, shape `[T, J, 13]`;
- the matching minipack `skeleton.json`;
- `anytop13_planetzoo_skinning_resources_v1/rigs/<object_name>/`;
- this repository and Cobra Tools; Blender 4.5 was used for validation.

The model normally sees `(motion - mean) / std`. Convert its output back to
raw values using the matching minipack `normalization.npz` before rendering.
The prediction must retain the exact minipack joint order and the raw channel
semantics. In particular, `3:9` is per-parent rot6d rather than the rotation
of the row's named joint.

## One Command

Run this with an ordinary Python interpreter that has NumPy. Replace the paths
with your local checkout.

```powershell
python tools/planetzoo/render_minipack_motion_to_ms2.py `
  --blender H:\blender4_5\blender.exe `
  --cobra-tools H:\path\to\cobra-tools `
  --resource-root H:\path\to\anytop13_planetzoo_skinning_resources_v1 `
  --object-name PZ_Bengal_Tiger_Male `
  --motion-path H:\generated\PZ_Bengal_Tiger_Male_motion_13ch.npy `
  --skeleton-path H:\minipack\clips\PZ_Bengal_Tiger_Male\skeleton.json `
  --output-root H:\renders\PZ_Bengal_Tiger_Male `
  --debug-frame-dir `
  --show-world-axes
```

The output root receives:

- `mesh_preview.mp4`: elevated three-quarter mesh preview; add
  `--show-world-axes` to draw scene axes (+X red, +Y green, +Z blue). The
  default camera is at -Y looking towards +Y, with the animal standing on XZ;
- `mesh_preview.blend`: inspectable Blender scene at frame one;
- `decoded_raw.bvh`: reconstructed BVH motion;
- `expanded_full_motion.npy`: the temporary full-topology AnyTop tensor;
- JSON reports and optional first/middle/last PNGs.

## What Is and Is Not Reconstructed

`expand_minipack_motion_to_full_rig.py` restores all full-rig channels needed
to decode the retained body hierarchy. It maps by joint name and broadcasts
the parent-indexed rot6d values to each original child slot. All 311 minipack
skeletons were checked to be induced subgraphs of their full rigs.

Joints deliberately absent from the minipack (face, tongue, ear details and
other terminal joints) stay in rest pose and follow their animated ancestor.
The mesh is therefore stable and animated, but those omitted independent
details cannot be recovered from a generated reduced-topology tensor.

The output scene uses a procedural rest-relative pose bridge instead of
copying local F-curves. Do not replace it with direct one-to-one rot6d-to-bone
assignment; AnyTop stores the rotation on child slots and Planet Zoo's `srb`
export helper otherwise leads to the known doubled global rotation error.
