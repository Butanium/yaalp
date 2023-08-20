# Ideas

## Maybe ?

- Random noise for actions too (already here for distance)
- Use the chosen direction to compute speed factor
- Yaal could learn their own "distance" mask which wouldn't have to be related to distance anymore. We could make them learn 1/4 of the mask to ensure symmetry.
- Plants could have some vision and choose their seeding direction according to what is around them

- Currently, the decisions give instantaneous actions, like speed and direction.
    -> In the future, maybe try to align the square with the direciton of the speed, and make the outputs accelerations rather than speeds.

- Plant reproduction : when eaten, planted on the way of the eater (co-evolution, agriculture, ...), otherwise, reproduce upon reaching some age.

- Food release upon death (predators, scavengers) (tune the amount of food based on the cost of fighting, living, reproducing, ..., and to mimic the flow of energy in real life (sun -> loss -> plant -> loss -> feminist -> loss -> predators))

## Todo

- Add bias for actions
- Add internal states as inputs
- Speed decision : grid * output / norm
- collision : when colliding with something, pushed at the boundary.
              Quadtree : try making a new one at each step, or dynamically updating the existing one by supression insertion of the moving object.

## Optimization (not for now)

- Parallelize creature evaluations (big matmul instead of a lot matmul). First we'd need to parallelize the evaluation of 1 creature I think

## rust
- use clippy (demander à pigeon)


