"""Logical update schedule for the authorized, matched 200-update pilot."""
from prrd_v3.contracts import require
from prrd_v3.render import Renderer

ACTIVATIONS = (1, 33, 65, 97, 129, 161)


class ScheduledRenderer(Renderer):
    def set_step(self, step):
        require(isinstance(step, int) and 1 <= step <= 200, 'Outside fixed 200-update schedule')
        self.active = sum(step >= s for s in ACTIVATIONS)
        for j, p in enumerate(self.residuals):
            p.requires_grad_(j < self.active)
