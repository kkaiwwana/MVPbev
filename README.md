# MVPbev

## Absrtact
This work aims to address the multi-view perspective RGB generation from text prompts given Bird-Eye-View(BEV) semantics. Unlike
prior methods that neglect layout consistency, lack the ability to
handle detailed text prompts, or are incapable of generalizing to
unseen view points, MVPbev simultaneously generates cross-view
consistent images of different perspective views with a two-stage
design, allowing object-level control and novel view generation at
test-time. Specifically, MVPbev firstly projects given BEV semantics to perspective view with camera parameters, empowering the
model to generalize to unseen view points. Then we introduce a
multi-view attention module where special initialization and denoising processes are introduced to explicitly enforce local consistency among overlapping views w.r.t. cross-view homography. Last
but not the least, MVPbev further allows test-time instance-level
controllability by refining a pre-trained text-to-image diffusion
model. Our extensive experiments on NuScenes demonstrate that
our method is capable of generating high-resolution photorealistic
images from text descriptions with thousands of training samples,
surpassing the state-of-the-art methods under various evaluation
metrics. We further demonstrate the advances of our method in
terms of generalizability and controllability with the help of novel
evaluation metrics and comprehensive human analysis. Our code
and model will be made available.