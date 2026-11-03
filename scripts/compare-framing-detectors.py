"""Neither framing detector works on turnaround REFERENCES. Measured.

SOMET-696 asked for a semantic framing check on `dir_<d>.png`. Two instruments
were scored against the same ten hand-labelled references (truth from opening
each PNG, not from a metric):

    framing.py CLIP bottom band   8/15, cut caught 0/7, whole passed 8/8
    Qwen3-VL-8B Q4_K_M + mmproj   8/15, cut caught 0/7, whole passed 8/8

8/15 is exactly the base rate of the whole class, so neither detector carries
information about the defect on this subject. Run with FRAMING_CHECK=1.

The VLM was given the WHOLE 512px image, which is the mistake framing.py had
already solved and documented: "scoring the whole cell measured 67.6% against
a 65% base rate - useless ... The view was the defect, not the model." It
answered feet=True, head_complete=True to all fifteen.

CLIP does carry partial signal and is the better starting point if this is
revisited: cut references scored 0.498, 0.528, 0.614, 0.635 against whole ones
at 0.981-1.000. But three cut references scored 0.896, 0.965 and 0.984, and
ALL THREE are the armoured Warrior concept, while the PROMPTS ("leather boots
and feet standing on the ground" vs "a robe hem and fabric, no feet visible")
were calibrated on a robed subject. Plate greaves are not a robe hem, so the
negative prompt does not describe this subject at all. That is the first thing
to try, not a new model.

WHAT ACTUALLY FIXED THE DEFECT was the camera: ISO_DISTANCE=wide. All five
wide references score 0.996-1.000 here and none is cut. Detector work is a
guard on a defect whose rate the camera change already collapsed.
"""

sys.path.insert(0, "/app")
os.environ["FRAMING_CHECK"] = "1"

from PIL import Image  # noqa: E402
import framing  # noqa: E402

IMAGES = "/app/images"

# (job, direction, expected_whole). Truth from opening each PNG.
CASES = [
    ("7f9fc8f1-cfa7-4f2d-9220-7809e7a629b1", "e", False),
    ("02cb757f-23fb-431b-a152-efe2a3f740d5", "e", False),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "e", False),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "w", False),
    ("67e00bd5-9516-4b2a-904a-dee207fb1302", "e", False),
    ("12b6be3a-ff7b-4ec5-aa47-4b5e06526dc8", "e", False),
    ("f11a661e-c8cc-46e0-a3b8-9bb799ed156b", "e", False),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "s", True),
    ("b0e77e2a-0e58-4c97-a8e9-72595a5b83c8", "n", True),
    ("02cb757f-23fb-431b-a152-efe2a3f740d5", "s", True),
    ("distwide8-0000-0000-0000-000000000000", "e", True),
    ("distwide8-0000-0000-0000-000000000000", "w", True),
    ("distwide8-0000-0000-0000-000000000000", "s", True),
    ("distwide8-0000-0000-0000-000000000000", "se", True),
    ("distwide8-0000-0000-0000-000000000000", "ne", True),
]

print(f"framing.py CLIP check, FRAMING_MIN={framing.FRAMING_MIN}, "
      f"band={framing.BAND}, zoom={framing.ZOOM}\n")
print(f"{'job':<10} {'dir':<4} {'truth':<6} {'score':>7} {'verdict':<7} ok")
right = cut_ok = cut_n = whole_ok = whole_n = 0
for job, d, expect in CASES:
    p = os.path.join(IMAGES, f"_job_{job}", f"dir_{d}.png")
    if not os.path.exists(p):
        print(f"{job[:8]:<10} {d:<4} {'-':<6} {'MISSING':>7}")
        continue
    s = framing.feet_in_frame(Image.open(p).convert("RGBA"))
    if s is None:
        print(f"{job[:8]:<10} {d:<4} {'-':<6} {'None':>7}  model unavailable")
        continue
    whole = not framing.is_close_up(s)
    hit = whole == expect
    right += hit
    if expect:
        whole_n += 1
        whole_ok += hit
    else:
        cut_n += 1
        cut_ok += hit
    print(f"{job[:8]:<10} {d:<4} {('whole' if expect else 'cut'):<6} "
          f"{s:>7.3f} {('whole' if whole else 'cut'):<7} "
          f"{'ok' if hit else '<-- WRONG'}")

print(f"\n{right}/{cut_n + whole_n} correct")
print(f"  cut references caught:   {cut_ok}/{cut_n}")
print(f"  whole references passed: {whole_ok}/{whole_n}")
