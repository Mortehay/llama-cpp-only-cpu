#!/usr/bin/env python3
"""Queue ONE real map through the real path.

    docker exec sprite_generator python /app/scripts/queue-a-map.py

WHY THIS EXISTS

No `kind='map'` job row has ever existed in this database. Every map built on
this box was built by calling worker functions directly - from smoke scripts
and one-off `docker exec` runs - because the HTTP endpoint needs a `generate`
scope and nobody was going to mint a key to talk to localhost.

The cost of that shortcut was invisible until it was not: `check-artifacts.py`
checks finished map ROWS, found none, and reported "maps checked, all whole"
on every run for a day (ADR 0008 D12). A whole half of the artifact check has
never had a subject.

So this queues a map the way the endpoint does - `maps.queue_map`, the same
validation and the same job row, minus only the scope check that HTTP adds -
and then there is something real to check.

It deliberately asks for the expensive version. Generated tiles, a region graph
with LLM naming, and scatter rules with `want` rather than `asset`, so the run
covers the painting, the tiles, the region routing and the pending-prop
resolver in one go. The LLM naming matters most: it is the call that makes
llama.cpp resident immediately before the painting, which is the exact
coexistence that broke the VAE decode.
"""

from __future__ import annotations

import sys

sys.path.insert(0, "/app")

import maps  # noqa: E402

# The map adapter, not the global default. The first run of this script left
# `llm_name` unset, so the painting came from `nerijs/pixel-art-xl` - the
# general pixel-art LoRA - and produced a map with 0.0% water on a theme that
# says "coast", plus stone as speckle rather than a range. The map adapter was
# trained on 114 world maps for exactly this step; at the same prompt it paints
# 20% water in coherent regions. Named explicitly because `default_model()` is
# global and knows nothing about maps.
MAP_MODEL = "stabilityai/stable-diffusion-xl-base-1.0+local:mapstyle"

SPEC = {
    "llm_name": MAP_MODEL,
    "name": "harbour-reach-3",
    "size": 24,
    "seed": 7,
    "prompt": "an island continent with a central mountain range, forests "
              "and a coast",
    "terrains": [
        {"id": 0, "name": "grass", "color": "#4a7c3f", "walkable": True,
         "prompt": "short green meadow grass"},
        # #39888f, NOT the #2850c8 the smoke-test set uses. MEASURED: the
        # painted water is a desaturated teal, and #2850c8 is 77.7 Lab away
        # from it while stone's mid-grey #6e6e73 is 26.0 - so a grey was the
        # nearest declared colour to every water pixel and quantisation, which
        # IS the tile-id binding, assigned the entire sea to stone. Two builds
        # warned "water covers 0.0%" before I stopped blaming the model.
        # Correcting this one value moves water 0.0% -> 20.0%. See ADR 0008 D15.
        {"id": 1, "name": "water", "color": "#39888f", "walkable": False,
         "prompt": "shallow blue sea water"},
        {"id": 2, "name": "sand", "color": "#c8be78", "walkable": True,
         "prompt": "pale coastal sand"},
        {"id": 3, "name": "stone", "color": "#6e6e73", "walkable": True,
         "prompt": "grey mountain rock"},
    ],
    # `want`, not `asset` - the point is to finish INCOMPLETE and let
    # `resolve_map_props` fill the gaps, which is the provisional state ADR
    # 0007 D7 exists for.
    "scatter": [
        {"layer": "props", "terrain": ["grass"], "want": "oak tree",
         "density": 0.06, "spacing": 2},
        {"layer": "props", "terrain": ["stone"], "want": "stone cairn",
         "density": 0.05, "spacing": 3},
        {"layer": "creatures", "terrain": ["grass"], "want": "grey wolf",
         "density": 0.02, "spacing": 4},
    ],
    "regions": 4,
    "theme": "a temperate coast with fishing towns",
    "region_llm": True,
}


def main() -> int:
    spec = maps.MapSpec(**SPEC)
    out = maps.queue_map(spec)
    for k, v in out.items():
        print(f"  {k}: {v}")
    print("\n  Poll: docker exec sprite_generator python "
          "/app/scripts/await-job.py " + out["job_id"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
