# cat_sitting_v1 reconstruction notes

Source: `cat_sitting_multiview_dataset_v1.zip`

## Inputs used

- 8 views: front, back, left, right, top, bottom, iso-left-top, iso-right-bottom
- 6 primary orthographic silhouette masks
- normalized landmarks
- relative dimensions: W=1.00, D=1.05, H=1.45
- source QA status: usable with warnings; isometric views are validation priors rather than hard geometry constraints

## Chosen output scale

- width: 100 mm
- depth: 105 mm
- height: 145 mm

This scale is arbitrary because the source dataset contains no physical metric calibration. The ratios follow the supplied relative dimensions.

## Construction strategy

1. Torso: multi-section elliptical loft, biased rearward at the lower body and forward at the shoulder/neck transition.
2. Head: overlapping ellipsoid, blended visually by intersection with the torso.
3. Ears: thin organic triangular extrusions on the XZ plane.
4. Hindquarters: paired ellipsoids.
5. Forelegs and paws: elongated/flattened ellipsoids.
6. Muzzle/cheeks: paired shallow ellipsoids plus a small nose volume.
7. Tail: circular sweep along a curved XZ spline, translated toward the rear of the body, with an ellipsoidal tip.
8. Fur, stripes, white chest/muzzle color regions, whiskers and paw-pad colors are intentionally omitted from STL geometry.

## Normalization

The assembled shape is normalized to approximately 100 x 105 x 145 mm through a final result transform. The STL may contain multiple overlapping shells because JSON CAD v1 uses a compound result for robust CI generation.

## Accuracy expectation

This is a coarse CAD-style organic reconstruction, not a neural image-to-mesh reconstruction. It should preserve the main seated-cat silhouette and part layout, while fine anatomical/fur details are intentionally simplified.
