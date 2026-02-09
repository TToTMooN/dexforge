# Task Assets

Phase-0 reorientation task assets live here.

- `cube.xml`: free-body cube used by both Allegro and xHand runs.

The robot is loaded from `src/dexplay/envs/robots/<robot>/model/*.xml`, and the task
attaches this shared cube entity so robot switching remains a config flag.
