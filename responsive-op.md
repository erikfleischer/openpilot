# segments
  * c0fe59cc1ebbd1b3/0000005b--8da62725eb/1 → short drive, segment 1 contains start
    behind lead vehicle

# todo
  * plot frequency response of existing filter in modeld.py for desired accelleration
  * design 2nd order bessel filter filter with the same dampening at 20 Hz and compare corner
    frequencies of both filters
  * design 4th order bessel filter filter with the same dampening at 20 Hz and compare corner
    frequencies of all three filters
  * add previews for both 2nd order and 4th order filter to jotPluggler which are applied
    to modelDebug/unfilteredDesiredAcceleration. Assume the filters are executed at 100 Hz
    sampling frequency.

# implementation notes
  * do not touch implementation filter in modeld.py → disable via setting
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
  * add 2nd order bessel filter with a calibratable corner frequency to LongitudinalPlanner:update() to filter. Initially set the corner frequency to 2 Hz.