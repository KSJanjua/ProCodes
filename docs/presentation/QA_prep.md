# Q&A Preparation — Instance-Aware Depth Estimation in Videos

> Likely questions for the final presentation, with short answers in plain
> words. Read them out loud once or twice — the answers are written the way
> you'd actually say them, not the way a paper would write them.

---

## 1. Big picture

**Q: In one line, what did you build?**
A system that takes a normal video and tells you, for every person, how far
away they are in metres — and keeps that answer correct when people overlap,
and stable from frame to frame.

**Q: Why is normal depth estimation not enough?**
Two reasons. Normal models give one depth map with no idea of "who" is in it —
overlapping people just blur together. And on video they flicker: the same
wall changes its depth value every frame. We needed per-person, stable,
metric depth.

**Q: What paper is this based on, and how far did you follow it?**
"Instance-Level Video Depth in Groups Beyond Occlusions", ICCV 2025. We
followed its three-phase recipe — scene depth, per-person masks and depth,
occlusion refinement. Then we went beyond it: the paper is single-frame, so
everything about time — memory, the flicker loss, identity across frames —
is ours.

**Q: What's the practical use of this?**
Anything that needs to know where people are in 3D from a normal camera:
activity monitoring, sports and group analysis, robots moving around people,
AR effects that need to know who is in front of whom.

**Q: What was the hardest part of the project?**
Honestly, the temporal part. Making depth stable over time is hard because a
single frame gives you nothing to anchor it to, and even the metric you use
to measure flicker is tricky — sensor noise can hide real improvements. It
took careful loss design and careful evaluation to get a clean gain.

---

## 2. The dataset

**Q: Why did you build your own dataset?**
Because none existed. The paper's dataset isn't public, and no public dataset
has per-person masks, identities, and metric depth together on video. We had
a ZED RGB-D camera, so we recorded raw RGB + depth and built all the
annotations ourselves.

**Q: How were the annotations created?**
Automatically, with a pipeline we wrote. A segmentation foundation model
(SAM) gives a mask for each person in each frame. We link the masks across
frames by overlap, so each person keeps one identity for the whole video.
Each person's ground-truth depth is the average of the sensor's depth values
inside their mask. When two masks fight over a pixel, the nearer person wins
it — because they are the one the camera actually sees.

**Q: How big is the dataset?**
Around 50,000 annotated frames in total — roughly 40,000 for training and
10,400 held out for testing. About 3,000 of the test frames contain real
person-on-person occlusion, which is the hard case we care about.

**Q: How do you know the automatic labels are good?**
Three safeguards: the depth comes straight from a calibrated sensor, not
from a model; very small or invalid instances are filtered out; and we
visually checked sequences with overlay videos. It's not hand-perfect, but
it's consistent, and the same rules apply to every frame.

**Q: What are the dataset's limits?**
One sensor, indoor-style scenes, depth capped at 10 metres, and only the
"person" class. Also, a single camera can't see behind people — so there is
no ground truth for hidden body parts. That shaped some of our choices later.

---

## 3. Phase 1 — scene depth

**Q: What does Phase 1 do?**
It paints the whole scene with depth: every pixel gets a value in metres.
It's the foundation the other phases build on.

**Q: What's the architecture, briefly?**
A strong pretrained vision backbone (the Depth Anything V2 encoder, which is
a DINOv2 ViT-Large), then a depth decoder that works from coarse to fine,
refining the depth in steps. Output is a full-resolution metric depth map.

**Q: Why not just use Depth Anything V2 directly?**
Two reasons. Its output is relative depth — it says "this is closer than
that" but not "this is 3.2 metres". We need real metres. And it has no idea
of people or occlusion, which is the whole point of our project. We reuse its
strong encoder, and train the rest for our task.

**Q: What loss did you train with?**
A scale-invariant log loss on depth (the standard one for metric depth,
from Eigen et al.). Our sensor gives true metric ground truth, so we train
directly against metres.

**Q: Why metric depth instead of relative?**
Because "the person is 3.2 metres away" is useful; "the person is 60% of the
scene depth" is not. Our sensor gives calibrated metres, so we can train for
the useful thing directly.

---

## 4. The temporal memory (our video extension)

**Q: What exactly is the memory module?**
A small recurrent block — a convolutional GRU — sitting inside Phase 1,
between the decoder and the final depth heads. It keeps a hidden state from
previous frames, so the model sees a bit of the past, not just the current
frame.

**Q: Why does it start "doing nothing"?**
Its output layer is initialised to zero, so on day one the model behaves
exactly like the per-frame model. Training then teaches it when the past is
worth using. That way it can only help — it never starts by hurting.

**Q: What loss makes the depth stable? Why not just compare frame t with frame t-1?**
That naive idea punishes real motion — a person walking toward the camera
*should* change depth. Our loss (temporal gradient matching, from the Video
Depth Anything line of work) compares the *change* in our prediction with
the *change* in ground truth. If the scene moves, we're allowed to move. Only
change that the ground truth doesn't have — flicker — gets punished.

**Q: How do you measure flicker?**
Temporal alignment error, TAE: the difference between the prediction's
frame-to-frame change and the ground truth's frame-to-frame change. Perfectly
tracking the real motion scores zero. Ours dropped from 0.059 to 0.055.

**Q: Couldn't the model just blur everything over time to look stable?**
That's the trap, and it's why accuracy matters: if it were just blurring,
per-frame accuracy would get worse. Ours got better at the same time —
error down from 0.082 to 0.072. So it learned real stability, not blur.

**Q: Why not use optical flow?**
Flow-based smoothing is heavy, adds another model, and fails exactly where we
care — on overlapping people. Our loss needs no flow at all, because the
dataset has per-frame ground truth to compare changes against.

**Q: Does the memory slow inference down?**
Barely. It's a tiny module (a couple of million parameters against a
350-million-parameter backbone) and it keeps constant state — one hidden
tensor — so it streams over any video length.

**Q: How did you train it on video?**
On short ordered clips. And one practical trick: our scenes move slowly, so
we sample clips where motion actually happens — the sampler measures motion
in the ground truth and prefers clips with movement. Otherwise the module
would mostly see frozen scenes and learn nothing.

---

## 5. Phase 2 — finding and following people

**Q: What does Phase 2 do?**
For each frame it outputs, per person: a segmentation mask, and one depth
number — "this person is at 3.2 metres."

**Q: What model is it?**
Mask2Former with a Swin-Large backbone, pretrained on COCO — a standard,
strong instance segmentation model. On top we added one small MLP head that
turns each detected person's internal feature vector into their depth value.

**Q: Why Mask2Former?**
It's the same family the paper uses, it's query-based — each detected person
is represented by one learned vector, which is exactly what we need for the
depth head and for identity — and pretrained weights are available.

**Q: How good is it?**
On the full test set: about 94.5% precision, 94% recall, 98% mask IoU, and
the per-person depth is off by about 6 cm on average. On the occlusion-heavy
frames it still holds 93% F1.

**Q: How do people keep the same identity across frames?**
Each detected person comes with a feature vector (their "query embedding").
The same person produces a similar vector in the next frame, even if the
detection order shuffles. So we match vectors between frames — that keeps one
identity per person, and even re-recognises someone after they were hidden
for a while. This idea comes from video instance segmentation research
(Video Mask2Former / MinVIS), but needs no video training at all.

**Q: Why not train a full video version of Mask2Former?**
Cost versus benefit. The full video model needs clip-level training and much
more memory, and would throw away our trained checkpoint. Research (MinVIS)
showed the per-frame model's own embeddings already carry identity — so we
get the benefit at basically zero cost.

**Q: Does Phase 2 work on data it wasn't trained on?**
On our data it's very strong. On very different footage it degrades — that's
normal for a model fine-tuned on one domain, and improving that is on our
future-work list.

---

## 6. Phase 3 — occlusion refinement

**Q: What does Phase 3 do?**
It fixes depth exactly where two people overlap. For every overlapping pair
it asks: who is in front, and by how much — then adjusts the depth in that
region.

**Q: How does it find the pairs?**
From Phase 2's output: confident detections whose boxes overlap. For each
such pair we cut out the region around both people, look at both together,
and predict a small correction for each one.

**Q: How is the correction applied?**
As a multiplicative factor on the base depth — roughly "make this person 3%
closer" — blended in softly inside the person's mask, so there are no hard
edges in the depth map. The correction is limited to about ±15%, so it can
sharpen things but can never produce a crazy value.

**Q: Why the ±15% limit?**
Because occlusion corrections are naturally small — people standing near
each other differ by tens of centimetres, not by half the scene. The limit
keeps the refinement gentle and artifact-free.

**Q: What did it improve?**
In the overlap regions — the exact pixels where two people cross — error
dropped from 0.059 to 0.055, and 98.6% of those pixels land in the tight
accuracy band. The rest of the scene stays at full quality, 0.078 overall.
The change happens only where it's needed.

**Q: The gain looks small. Is it worth it?**
The number is small because overlap pixels are a small fraction of the image
and the base is already strong there. But those are the most important pixels
for this task — that's precisely where every other model fails — and the
qualitative difference at boundaries is clearly visible.

**Q: How do you supervise depth for hidden body parts?**
We can't — a single camera has no ground truth behind an occluder. So
training supervises only visible pixels. For hidden moments, our identity
memory holds the person's last reliable depth until they reappear.

**Q: Why compare boxes and not masks for overlap?**
Because our ground-truth masks are "visible-only": when one person hides
another, their masks don't share pixels by definition, so mask overlap is
always near zero. Boxes still overlap, so boxes tell us who might be
occluding whom.

---

## 7. Results & metrics

**Q: Explain your metrics in one line each.**
- **Abs Rel** — average error as a fraction of true depth. 0.072 means ~7% off.
- **RMS** — average error in metres, punishing big mistakes more.
- **Threshold accuracy (σ1)** — share of pixels within 25% of the truth.
- **TAE** — how much our frame-to-frame change differs from the true change; the flicker score.
- **F1 / IoU** — detection and mask quality for the people.

**Q: Your headline numbers?**
Scene depth with memory: 0.072 relative error, 94.1% accuracy, RMS 0.374,
flicker (TAE) 0.055 — every one better than the per-frame baseline. People:
~94% precision/recall, 98% IoU, 6 cm per-person depth error. Overlap regions:
0.055 error, 98.6% accuracy.

**Q: How does this compare to the paper?**
Not directly comparable — the paper reports on its own dataset, we report on
ours, and the scenes differ. Within our data, we compare against our own
per-frame baseline, which is the fair comparison, and everything improved.

**Q: Was the test set really held out?**
Yes — split by sequence, so the test videos were never seen in training, and
all numbers are on the full 10,400-frame test split, not a subset.

---

## 8. Engineering & demos

**Q: How do the demo videos get their stable colours?**
From the identity tracking: each person's ID maps to a fixed colour. Since
IDs persist across frames (embedding matching), the colours do too.

**Q: Does it run on arbitrary videos, outside your dataset?**
Yes, the whole pipeline runs on any video file. Quality is best in-domain;
for very different footage we also have inference-time helpers — a depth
colour-range tool and an optional temporal smoother — for clean demos.

**Q: How fast is it?**
It's a research pipeline, not real-time: two large backbones run per frame.
The temporal parts add almost nothing on top. Speed was not a goal this
round; distillation or lighter backbones would be the route if it becomes one.

**Q: How is the code organised?**
Two packages: `instancedepth` is the paper reproduction — dataset engine,
three phases, training and evaluation. `videodepth` is everything we added
for video — the memory module, the temporal loss, identity tracking, the
improved occlusion head — plus the tools that make the videos. About 140
automated tests cover the pieces.

**Q: How reproducible is this?**
One script runs the whole chain — training, evaluation, in order — with
fixed seeds, config files for every run, and a manifest saved with each
experiment recording exactly what produced it.

---

## 9. Limitations & future work

**Q: Main limitations?**
One camera, one indoor domain, one class (person), depth up to 10 metres, no
ground truth behind occluders, and not real-time. All known, all shaping the
future-work list.

**Q: What's next?**
Three things. Test the occlusion refinement on other occlusion-heavy
datasets, to see whether the gains transfer. Make the whole system hold up on
footage from any camera. And release our dataset as a public benchmark —
nothing like it exists publicly — together with the write-up.

**Q: What would you do differently with more time?**
Spend more of it on cross-domain robustness earlier. In-domain results came
together well; making the same quality appear on any random video is the
bigger open problem, and it deserves the most attention next.

---

## 10. Possible curveballs

**Q: Isn't this just three existing models glued together?**
The components are deliberately standard — that's good engineering. What's
new is everything between them: the dataset that trains them, the temporal
memory and loss, identity across frames, the pair-wise occlusion head, and
making all of it agree on one consistent answer per person per frame.

**Q: Could the memory hurt when the scene changes suddenly?**
It's a learned gate, not a fixed average — it learned when to trust the past.
On cuts or fast motion it leans on the current frame; the loss never rewarded
blind smoothing.

**Q: What happens with more people, say ten?**
Phase 2 detects each independently, so detection scales fine. Phase 3 looks
at overlapping pairs, and the number of pairs grows — but only truly
overlapping, confident pairs are processed, so in practice it stays small.

**Q: What if two people wear the same clothes — does identity confuse them?**
The embeddings encode more than colour — position, pose, context — and we
also weigh in spatial overlap between frames. Identical twins crossing in
identical clothes is genuinely hard for any tracker; for normal cases it
holds.

**Q: Does a moving camera break it?**
The depth and masks are per-frame, so they're fine. The temporal parts
assume some continuity, and our training data is from a static camera — so a
fast-moving camera is out-of-domain and part of the robustness work ahead.

**Q: Why should we trust auto-generated ground truth?**
The depth itself is measured by a calibrated sensor — that part isn't
generated. Only the masks and identities are automatic, from a
state-of-the-art segmenter, with filtering and visual checks. And every model
is judged against the same labels, so comparisons stay fair.
