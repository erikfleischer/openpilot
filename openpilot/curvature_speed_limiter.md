# goals
  * limit ego velocity based on predicted curvature

# available data
  * modelV2.velocity.x — predicted longitudinal speed (m/s)
  * modelV2.orientationRate.z — predicted yaw rate (rad/s)
  * curvature via bicycle model: κ = |ψ̇| / max(|v_x|, MIN_SPEED)

# outputs to modify
  * mpc long planner uses a cruise obstacle to set cruise speed
  * Based on predicted curvature and lateral accelleration limit a speed limit can be estimated over time. This limit can be applied to v_cruise_clipped in LongitudinalMpc::update
  * lateral accelleration limit should be set based on driving personality and limited by ISO safety limits
  * limited speed should be forward and backward propagated
  * current speed will always lag behind the set speed. This needs to
    taken into account while back propagating the speed limit.
