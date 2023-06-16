from dm_control import suite, viewer
from npc import NPC, Controller

env   = suite.load(domain_name="cartpole", task_name="balance")
dt    = env.control_timestep()
print("dt =", dt)

class CartpoleBalanceController(Controller):
    def advance(self, position, angle, velocity, angular_velocity):
        a = 1.0
        p = +.01
        v = -.00
        w = +.01
        sp = 0
        cmd = -sp + a * angle + p * position + v * velocity + w * angular_velocity
        cmd = max(-5, min(+5, cmd))
        return [cmd]

npc = NPC(CartpoleBalanceController(),
        iterations                  = 1,
        num_steps                   = 1,
        sense_spec                  = [(-2, +2, 0.01), (-1, +1, 0.01), (-10, +10, 0.01), (-10, +10, 0.01),],
        action_spec                 = [(-5, +5, 0.1)],
        input_num_active            = 5,
        output_num_active           = 5,
        granule_num_cells           = 10000,
        granule_num_active          = 50,
        granule_threshold           = .2,
        granule_potential_pct       = .25,
        granule_learning_period     = 100,
        granule_num_patterns        = 5,
        granule_weight_gain         = 5,
        granule_boosting_period     = 10000,
        purkinje_threshold          = .1,
        purkinje_potential_pct      = .25,
        purkinje_learning_period    = 100,
        purkinje_num_patterns       = 100,
        purkinje_weight_gain        = 1.5,
        seed                        = None,)

def control_policy_wrapper(time_step):
    if time_step.first():
        npc.reset()

    pos, sin, cos = time_step.observation['position']
    vel, ang = time_step.observation['velocity']

    cmd = npc.advance(pos, cos, vel, ang)[0]

    return [cmd]

def test_cartpole_balance():
    # Run headless and verify performance.
    scores = []
    for trial in range(3):
        time_step = env.reset()
        scores.append(0)
        while not time_step.last():
            angle = time_step.observation['position'][2]
            if angle > .66 or -.66 > angle:
                break
            action = control_policy_wrapper(time_step)
            time_step = env.step(action)
            scores[-1] += dt
        print(scores[-1])
    assert all(x > 9.9 for x in scores)

if __name__ == '__main__':
    viewer.launch(env, policy=control_policy_wrapper)
