import unittest

from hw.device import create_device, launch
from hw.isa import parse_asm

# A short ALU-heavy kernel.  The adds are independent so the scheduler stalls
# mainly on ALU issue latency rather than register dependencies.
ALU_BOUND_ASM = """
MOV R0, 0
MOV R1, 1
ADD R10, R0, R1
ADD R11, R0, R1
ADD R12, R0, R1
ADD R13, R0, R1
ADD R14, R0, R1
EXIT
"""


# The same setup followed by four independent global loads.  Each load occupies
# the warp for the global-memory latency, so this kernel is clearly
# memory-bound when compared with the ALU kernel above.
MEMORY_BOUND_ASM = """
MOV R0, 0
LDG R1, [R0]
LDG R2, [R0]
LDG R3, [R0]
LDG R4, [R0]
EXIT
"""


# A tight read-after-write chain.  The issue latency alone cannot hide the
# dependency, so the profile should expose scoreboard stalls.
SCOREBOARD_ASM = """
ADD R0, R0, 1
ADD R0, R0, 1
ADD R0, R0, 1
EXIT
"""


# Warp 0 skips a delay block and reaches the barrier first.  Warp 1 performs
# four global loads before arriving, which leaves warp 0 blocked at BAR.SYNC
# long enough for the profile to report barrier stalls.
ASYMMETRIC_BARRIER_ASM = """
MUL R0, tid.x, 4
STS [R0], tid.x
MOV R1, warpid
CMP.EQ R2, R1, 0
BRA fast, R2
LDG R3, [R0]
LDG R4, [R0]
LDG R5, [R0]
LDG R6, [R0]
BRA barrier
fast:
barrier:
BAR.SYNC
MOV R7, ntid.x
SUB R8, R7, tid.x
SUB R8, R8, 1
MUL R8, R8, 4
LDS R9, [R8]
STG [R0], R9
EXIT
"""


# All lanes write to shared-memory address zero, so every active lane hits the
# same 4-byte bank.  This is the pathological case used to exercise bank
# conflict detection.
BANK_CONFLICT_ASM = """
MUL R0, tid.x, 0
MOV R1, tid.x
STS [R0], R1
EXIT
"""


class TestTimingModelEndToEnd(unittest.TestCase):
    """Step 004 profile-mode tests."""

    def _launch_profile(self, asm: str, block_dim: int = 8):
        device = create_device(global_size=1 << 20, sm_count=4)
        profile = launch(
            device,
            parse_asm(asm),
            grid_dim=(1,),
            block_dim=(block_dim,),
            profile=True,
        )
        self.assertIsNotNone(profile)
        return device, profile

    def test_profile_mode_is_deterministic(self):
        snapshots = [
            self._launch_profile(MEMORY_BOUND_ASM)[1].as_dict() for _ in range(3)
        ]
        self.assertEqual(snapshots[0], snapshots[1])
        self.assertEqual(snapshots[1], snapshots[2])

    def test_memory_bound_differs_from_alu_bound(self):
        _, alu_profile = self._launch_profile(ALU_BOUND_ASM)
        _, memory_profile = self._launch_profile(MEMORY_BOUND_ASM)

        self.assertGreater(memory_profile.stalls.memory, alu_profile.stalls.memory)
        self.assertGreater(alu_profile.stalls.alu, memory_profile.stalls.alu)
        self.assertGreater(memory_profile.global_load_bytes, 0)

    def test_register_dependency_shows_scoreboard_stalls(self):
        _, profile = self._launch_profile(SCOREBOARD_ASM)
        self.assertGreater(profile.stalls.scoreboard, 0)

    def test_barrier_stalls_are_counted(self):
        device, profile = self._launch_profile(ASYMMETRIC_BARRIER_ASM, block_dim=64)
        self.assertGreater(profile.stalls.barrier, 0)

        expected = [float(i) for i in reversed(range(64))]
        self.assertEqual(device.global_memory.read_floats(0, 64), expected)

    def test_shared_bank_conflicts_are_reported(self):
        _, profile = self._launch_profile(BANK_CONFLICT_ASM)
        self.assertGreater(profile.shared_bank_conflicts, 0)
        self.assertGreater(profile.shared_store_bytes, 0)


if __name__ == "__main__":
    unittest.main()
