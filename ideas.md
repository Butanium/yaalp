# Ideas

## Maybe ?

- Random noise for actions too (already here for distance)
- Use the chosen direction to compute speed factor
- Yaal could learn their own "distance" mask which wouldn't have to be related to distance anymore. We could make them learn 1/4 of the mask to ensure symmetry.
- Plants could have some vision and choose their seeding direction according to what is around them

-

## Todo

- Add bias for actions
- Add internal states as inputs

## Optimization (not for now)

- Parallelize creature evaluations (big matmul instead of a lot matmul). First we'd need to parallelize the evaluation of 1 creature I think

## rust
- use clippy (demander à pigeon)