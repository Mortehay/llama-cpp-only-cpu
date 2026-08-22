"""Build a crude but unambiguous box humanoid, as a CONTROL for the rig stage.

WHY
Four TripoSR-derived characters have gone through UniRig and only one produced a
skeleton whose limb ends could be labelled. That leaves two very different
diagnoses with very different fixes:

  * UniRig is unreliable on this kind of input      -> change the rigger
  * TripoSR meshes are unsuitable for UniRig        -> change image-to-3D, or
                                                       the source pose

A mesh with no ambiguity at all separates them. This is a T/A-posed figure made
of boxes: head, torso, two arms and two legs, all clearly separated, obviously
symmetric, watertight, and with the same Z-up convention and roughly the same
proportions as the TripoSR output so nothing else varies.

If UniRig cannot produce a clean 5-limb symmetric skeleton for THIS, the rigger
is the problem and no amount of better reconstruction will save the route. If it
can, the meshes are the problem and the fix is upstream.

Run:  python make_test_humanoid.py /app/images/control.obj
"""
import sys


def box(bpy, name, centre, size):
    # size=1 already gives an edge length of 1, so scale IS the final dimension.
    # Halving it here (the obvious-looking "radius" reading) silently produced a
    # figure at half the intended depth.
    bpy.ops.mesh.primitive_cube_add(size=1, location=centre)
    ob = bpy.context.active_object
    ob.name = name
    ob.scale = size
    return ob


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "/app/images/control.obj"
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)

    # Proportions roughly matching the TripoSR characters: ~1.0 tall, depth
    # smaller than width, limbs held clear of the torso so there is no question
    # of them being fused.
    parts = [
        ("torso", (0.00, 0.0, 0.62), (0.26, 0.14, 0.40)),
        ("head",  (0.00, 0.0, 0.92), (0.18, 0.16, 0.20)),
        ("arm_L", (-0.26, 0.0, 0.62), (0.26, 0.10, 0.10)),
        ("arm_R", (0.26, 0.0, 0.62), (0.26, 0.10, 0.10)),
        ("leg_L", (-0.10, 0.0, 0.21), (0.11, 0.11, 0.42)),
        ("leg_R", (0.10, 0.0, 0.21), (0.11, 0.11, 0.42)),
    ]
    objs = [box(bpy, n, c, s) for n, c, s in parts]

    for ob in objs:
        ob.select_set(True)
    bpy.context.view_layer.objects.active = objs[0]
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    bpy.ops.object.join()
    merged = bpy.context.active_object

    # Fuse the interpenetrating boxes into ONE closed surface. remove_doubles
    # cannot do this - the boxes share no vertices, they merely overlap, so it
    # welds nothing and leaves six separate shells. A voxel remesh rebuilds the
    # union as a single manifold, which is what the TripoSR meshes are; leaving
    # six components would make the control differ from the thing it is meant to
    # be a control for.
    rm = merged.modifiers.new(name="remesh", type="REMESH")
    rm.mode = "VOXEL"
    rm.voxel_size = 0.012
    bpy.ops.object.modifier_apply(modifier=rm.name)

    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    bpy.ops.wm.obj_export(filepath=out, export_selected_objects=True,
                          export_materials=False)
    dims = merged.dimensions
    print(f"wrote {out}")
    print(f"  verts {len(merged.data.vertices)}  faces {len(merged.data.polygons)}")
    print(f"  dimensions  X {dims[0]:.3f}  Y {dims[1]:.3f}  Z {dims[2]:.3f}")
    print("  limbs are separated by construction; Z-up, like the TripoSR output")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
