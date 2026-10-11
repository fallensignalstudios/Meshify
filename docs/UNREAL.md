# Getting Meshify assets into Unreal Engine 5

## Quick path: drag in the GLB

1. Drag `SM_<name>.glb` into the Content Browser. UE5's Interchange glTF
   importer creates the static mesh, the material and the textures.
2. Scale is already right: Meshify exports metres (glTF standard), and Unreal
   converts to centimetres. If you set **Height (m)** in Meshify, the asset is
   that tall in-engine.
3. The pivot is at the bottom centre, so it sits on the floor when you drop it in.

## Best quality: build the material from the texture set

The `textures/` folder follows Unreal naming conventions:

| File | Texture settings | Material pin |
|---|---|---|
| `T_<name>_BaseColor.png` | sRGB **on** | Base Color |
| `T_<name>_Normal_DX.png` | Compression: **Normalmap** | Normal |
| `T_<name>_ORM.png` | sRGB **off**, Compression: **Masks** | R → Ambient Occlusion, G → Roughness, B → Metallic |

Use `Normal_DX` in Unreal (DirectX, green flipped). `Normal_GL` is for
Blender, Godot and glTF tools.

Tip: make one master material with these three texture parameters, then
create a Material Instance per asset. Importing a new asset is then just
swapping three textures.

## Dense meshes and Nanite

If you chose **Full (Nanite)** as the face budget, open the static mesh and
tick **Enable Nanite Support**. For non-Nanite props, use 10k-50k faces and
let Unreal auto-generate LODs (Static Mesh Editor → LOD Settings → Number of LODs).

## Making it look real in-engine

- Dial **Remove baked lighting** up (0.7-0.9) if the asset looks "pre-lit"
  under your scene lighting, or down if it loses too much colour variation.
- Metallic defaults to 0. For metal parts, paint the ORM blue channel or
  override Metallic in the material instance.
- Real-world height matters more than you'd think for realism. A barrel is
  about 0.9 m, a door 2.1 m, a person 1.75 m.
