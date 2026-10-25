# Test image sources

All images are crops/derivatives made by `scripts/make_test_assets.py` (YOLO11n person boxes, hand-selected).

- COCO val2017 (http://images.cocodataset.org/val2017/<id>.jpg): image annotations CC BY 4.0, images from Flickr under
  various CC licences (see COCO `licenses`); used here only as small crops for offline testing. Ids used:
  785, 872, 885, 1000, 1296, 2473, 2532, 3156, 4134, 4395, 5001, 5060, 5193 (persons); 1000-independent
  non-person crops come from the ultralytics `bus.jpg` / `zidane.jpg`.
- ultralytics assets `bus.jpg`, `zidane.jpg` (AGPL-3.0 repo sample images, https://github.com/ultralytics/ultralytics).
  p16, p17, p18, n01, n02, n03 derive from these.
- Crop weights: yolo11n.pt (ultralytics, AGPL-3.0), run on CPU.
- Aerial-like crops `a01..a06`: downscaled to 90 px height, upscaled (bicubic) to >=224 px short side.
- Photos of real people, for local testing only; do not redistribute outside the repo.
