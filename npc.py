import htm_rs
import copy
import math

class Controller:

    def clone(self):
        return copy.deepcopy(self)

    def reset(self):
        pass

    def advance(self, **senses) -> 'commands':
        raise NotImplementedError()

class NPC:
    def __init__(self, controller, iterations, sense_spec, action_spec, **parameters):
        sense_spec  = list(sense_spec)
        action_spec = list(action_spec)
        self.controller = controller
        self.iterations = int(iterations)
        self.cerebellum = htm_rs.Cerebellum(
                input_spec = sense_spec + action_spec,
                output_spec = sense_spec,
                **parameters)
        assert isinstance(self.controller, Controller)
        assert self.iterations >= 0

    def reset(self):
        self.controller.reset()
        self.cerebellum.reset()

    def advance(self, *senses):
        # 
        commands = tuple(self.controller.advance(*senses))
        for iteration in range(self.iterations):
            commands = self._adjust(senses, commands, learn=(iteration==0))

        return commands

    def _adjust(self, senses, commands, learn):
        # 
        next_senses = self.cerebellum.advance(senses + commands, senses if learn else None)
        if any(math.isnan(x) for x in next_senses):
            return commands

        next_commands = tuple(self.controller.advance(*next_senses))

        # Add the adjustment to the original commands.
        return tuple(a + b for a, b in zip(commands, next_commands))

    def __str__(self):
        return "NPC " + str(self.cerebellum)

