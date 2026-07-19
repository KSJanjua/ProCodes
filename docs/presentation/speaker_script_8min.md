# Speaker Script — 8 minutes, simple spoken style

> Matches the final 12-slide deck (Final_PPT.pptx). ~1,050 words at a relaxed
> ~135 wpm. Keep the pauses on slides 9 and 11 — the visuals do the talking.

## SLIDE 1 — Title (0:00 – 0:20)
Good afternoon everyone. I'm Karanvir, and this is the final presentation of
my internship with the Visual Intelligence team. My project is about depth
estimation — but not the usual kind. I wanted depth for every single person
in a video, and I wanted it to stay stable over time. Let me show you what I
mean.

## SLIDE 2 — The Problem (0:20 – 1:05)
So, depth estimation today is actually pretty good. Models like DepthPro and
Depth Anything can give you a really detailed depth map from one image. But
the moment I ran them on videos of people, two problems showed up.

First — these models don't know what a person is. Everything is just pixels.
If two people stand behind each other, their depths blur together, and you
can't say who is where.

Second — they get shaky on video. Run them frame by frame and the depth keeps
jumping. Even a wall that isn't moving changes its value between frames. It
looks like flickering, and you can't build anything reliable on top of that.

That was my starting point.

## SLIDE 3 — The Goal (1:05 – 1:40)
I set four simple targets.

One — depth for each person. Not one map for the whole scene.
Two — it should stay correct when people overlap. Front person closer, back
person farther, clean boundary between them.
Three — steady across frames. If someone stands still, their depth should
stand still too.
Four — real metres. If the model says 3.2 metres, that person is really 3.2
metres away.

If all four work, we have something genuinely useful.

## SLIDE 4 — The Foundation (1:40 – 2:30)
But before training anything, I hit a wall. There is no dataset for this
task. The paper I followed used their own data, and it's not public. All I
had was one ZED camera — RGB frames and raw depth. That's it. No masks, no
identities, no labels.

So I built the dataset myself. On the left is what I started with — just
frames and depth. On the right is what came out of it: a mask for every
person, an identity that stays with them through the whole video, and a
ground-truth depth value for each person. Segmentation came from SAM, and
identities were linked across frames by matching overlaps.

In the end — ten thousand four hundred annotated frames, and about three
thousand of them have real occlusions. Everything else in this project
stands on this dataset.

## SLIDE 5 — Our Idea (2:30 – 3:20)
The method has three phases, following the ICCV 2025 paper — plus one thing
that's ours.

Phase one paints the scene. Every pixel gets a metric depth.
Phase two finds each person — a mask, and one depth number per person.
Phase three fixes the overlaps. When two people cross, it looks at that pair
and corrects the depth right there.

And then the part we added: memory. The paper works one frame at a time. We
let the model remember what it saw in previous frames. That's our video
extension — and honestly, it turned out to be the most important piece.

## SLIDE 6 — Phase 1: Scene Depth + Memory (3:20 – 4:20)
Let's look at Phase 1. The backbone is the Depth Anything V2 encoder, with
our depth decoder on top. First we trained it purely frame by frame — that's
the baseline column.

Then we added the memory module. It's a small recurrent block that carries
information from the last few frames. It starts off doing nothing, and
during training it slowly learns when the past is worth trusting.

And every number got better. RMS went from 0.407 down to 0.374. Relative
error from 0.082 to 0.072. Our flicker measure, TAE, from 0.059 to 0.055.
And accuracy went up, from about 93 to 94 percent. Normally you trade
accuracy for stability. Here we got both at the same time.

## SLIDE 7 — Phase 2: Finding and Following People (4:20 – 5:10)
Phase 2 is about the people. I used Mask2Former, pretrained on COCO — a
strong segmentation model — and added one small head on top that gives each
person a single depth number.

So for every frame we get: a mask per person, and a label like "this person
is at 3.2 metres."

The numbers are solid. Around 94 and a half percent precision, 94 recall,
and 98 percent IoU on the masks. And on the hard frames — the ones where
people actually overlap — it still holds 93 percent F1. The per-person depth
is off by only about six centimetres on average. That's a good base for
phase 3.

## SLIDE 8 — Phase 3: Occlusion-Aware Refinement (5:10 – 6:05)
Phase 3 deals with the overlaps themselves. We take the dense depth and the
masks, and for every pair of overlapping people, a small head asks one
question: who is in front, and by how much. Then it nudges the depth — only
in that overlap region.

On the three thousand occlusion frames, depth error in those regions dropped
from 0.059 to 0.055, and 98.6 percent of the overlap pixels land inside the
tight accuracy band.

And the important part — the rest of the scene doesn't move. Still 0.078
overall. The correction happens exactly where it's needed, and nowhere else.

## SLIDE 9 — Results (6:05 – 6:40)
Here's the system across different kinds of scenes. A crowded group. Two
people crossing — that one is the classic identity killer. Someone close to
the camera. A wide scene with small people.

Same model, same settings, nothing tuned per scene. The boundaries stay
clean, each person keeps their colour, and the depth stays calm.

## SLIDE 10 — Beyond the Paper (6:40 – 7:35)
So what in this project is actually ours, beyond the paper? Four things.

The dataset — built from zero, from nothing but raw RGB and depth.

The video part — the paper is single-frame. We added the memory, and a loss
that punishes flicker but never punishes real motion. That's what made the
depth stable.

Identity — one person, one identity, held through occlusions. When someone
is hidden, we keep their depth from the last time we saw them, instead of
starting over.

And a better occlusion head — corrections are limited to about fifteen
percent, so it can help but it can never break the image. That's where the
98.6 percent comes from.

## SLIDE 11 — Demo (7:35 – 7:50)
Enough numbers. Here's everything running together in one clip — depth,
masks, per-person metres, all at once. Keep an eye on the moment the two
people cross.

(let the clip play)

## SLIDE 12 — Thank You (7:50 – 8:00)
That's my project. Thank you — happy to take any questions.
